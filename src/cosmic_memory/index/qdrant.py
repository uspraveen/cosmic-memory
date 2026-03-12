"""Qdrant-native passive recall backend aligned with Cosmic's hybrid retrieval shape."""

from __future__ import annotations

import asyncio
import inspect
from importlib.metadata import version
import re
from time import perf_counter
import uuid
from collections.abc import Sequence

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    CanonicalMemorySnapshot,
    GenerateEmbeddingsRequest,
    IndexPointState,
    PassiveRecallDiagnostics,
    PassiveRecallRequest,
    PassiveRecallResponse,
    RecallItem,
)
from cosmic_memory.embeddings.base import EmbeddingService
from cosmic_memory.index.sparse import SparseEncoder
from cosmic_memory.retrieval import (
    passive_candidate_limit,
    recall_item_from_payload,
    rerank_passive_items,
    select_passive_items,
)
from cosmic_memory.storage.markdown_store import canonical_record_hash


class QdrantHybridMemoryIndex:
    """Qdrant-native hybrid passive recall backend."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        sparse_encoder: SparseEncoder | None = None,
        sparse_model_name: str = "Qdrant/bm25",
        collection_name: str = "memories",
        vector_size: int | None = None,
        url: str | None = None,
        path: str | None = None,
        embed_batch_size: int = 128,
        embed_parallel_requests: int = 4,
        dense_encoding_format: str = "base64_int8",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantHybridMemoryIndex. "
                "Install project dependencies with `python -m pip install -e .`."
            ) from exc

        self._models = self._import_models()
        self.client = QdrantClient(url=url, path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size or embedding_service.dimensions
        self.embedding_service = embedding_service
        self.sparse_encoder = sparse_encoder
        self.sparse_model_name = sparse_model_name
        self.embed_batch_size = embed_batch_size
        self.embed_parallel_requests = embed_parallel_requests
        self.dense_encoding_format = dense_encoding_format
        self._is_local_path = path is not None and url is None
        self._client_lock = asyncio.Lock()
        self._ready = False

        if self.sparse_encoder is None and not _supports_qdrant_native_bm25():
            raise RuntimeError(
                "Qdrant-native BM25 requires qdrant-client>=1.15.2. "
                "Upgrade qdrant-client or inject an explicit SparseEncoder fallback."
            )
        if self.sparse_encoder is None and path is not None and not _has_fastembed():
            raise RuntimeError(
                "Local-path Qdrant native BM25 requires fastembed in the client environment. "
                "Install `fastembed` or inject an explicit SparseEncoder fallback."
            )

    async def ensure_ready(self) -> None:
        if self._ready:
            return

        if self._is_local_path:
            async with self._client_lock:
                if self._ready:
                    return
                await self._ensure_ready_unlocked()
                self._ready = True
            return

        await self._ensure_ready_unlocked()
        self._ready = True

    async def _ensure_ready_unlocked(self) -> None:
        exists = False
        if hasattr(self.client, "collection_exists"):
            exists = await asyncio.to_thread(self.client.collection_exists, self.collection_name)
        if exists:
            return

        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=self.collection_name,
            vectors_config={
                "dense": self._models.VectorParams(
                    size=self.vector_size,
                    distance=self._models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": self._models.SparseVectorParams(
                    modifier=self._models.Modifier.IDF,
                ),
            },
        )

    async def sync_record(self, snapshot: CanonicalMemorySnapshot) -> None:
        await self.sync_records([snapshot])

    async def sync_records(self, records: Sequence[CanonicalMemorySnapshot]) -> None:
        if not records:
            return

        await self.ensure_ready()
        contents = [snapshot.record.content for snapshot in records]
        if self.sparse_encoder is not None:
            dense_task = asyncio.create_task(
                self.embedding_service.generate(
                    GenerateEmbeddingsRequest(
                        texts=contents,
                        dimensions=self.vector_size,
                        batch_size=min(self.embed_batch_size, max(len(contents), 1)),
                        max_parallel_requests=self.embed_parallel_requests,
                        encoding_format=self.dense_encoding_format,
                        normalize=True,
                    )
                )
            )
            sparse_task = asyncio.create_task(self.sparse_encoder.encode(contents))
            dense_result, sparse_vectors = await asyncio.gather(dense_task, sparse_task)
        else:
            dense_result = await self.embedding_service.generate(
                GenerateEmbeddingsRequest(
                    texts=contents,
                    dimensions=self.vector_size,
                    batch_size=min(self.embed_batch_size, max(len(contents), 1)),
                    max_parallel_requests=self.embed_parallel_requests,
                    encoding_format=self.dense_encoding_format,
                    normalize=True,
                )
            )
            sparse_vectors = None

        dense_vectors = [item.vector for item in sorted(dense_result.items, key=lambda item: item.index)]

        points = []
        for index, (snapshot, dense_vector) in enumerate(zip(records, dense_vectors, strict=True)):
            record = snapshot.record
            payload = {
                "memory_id": record.memory_id,
                "type": record.kind.value,
                "status": record.status.value,
                "version": record.version,
                "title": record.title,
                "tags": record.tags,
                "path": snapshot.path,
                "updated_at": record.updated_at.isoformat(),
                "created_at": record.created_at.isoformat(),
                "content_hash": snapshot.content_hash or canonical_record_hash(record),
                "token_count": snapshot.token_count,
                "content": record.content,
                "source_kind": record.provenance.source_kind if record.provenance else None,
                "confidence": record.metadata.get("confidence"),
                "canonical_key": record.metadata.get("canonical_key"),
                "always_include": record.metadata.get("always_include"),
                "supersedes": record.supersedes,
            }
            points.append(
                self._models.PointStruct(
                    id=_point_id_for_memory_id(record.memory_id),
                    vector=self._build_point_vectors(
                        dense_vector=dense_vector,
                        content=record.content,
                        sparse_vector=None if sparse_vectors is None else sparse_vectors[index],
                    ),
                    payload=payload,
                )
            )

        await self._run_client_call(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points,
        )

    async def snapshot(self) -> dict[str, IndexPointState]:
        await self.ensure_ready()
        offset = None
        states: dict[str, IndexPointState] = {}

        while True:
            records, next_offset = await self._run_client_call(
                self.client.scroll,
                collection_name=self.collection_name,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                memory_id = str(payload.get("memory_id") or record.id)
                raw_status = payload.get("status")
                states[memory_id] = IndexPointState(
                    memory_id=memory_id,
                    point_id=str(record.id),
                    status=RecordStatus(raw_status) if raw_status else None,
                    content_hash=payload.get("content_hash"),
                )
            if next_offset is None:
                break
            offset = next_offset

        return states

    async def delete_records(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        await self.ensure_ready()
        snapshot = await self.snapshot()
        target_ids = set(memory_ids)
        point_ids = [
            entry.point_id
            for memory_id, entry in snapshot.items()
            if memory_id in target_ids and entry.point_id is not None
        ]
        if not point_ids:
            return
        await self._run_client_call(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=point_ids,
            wait=True,
        )

    async def reset(self) -> None:
        await self._run_client_call(
            self.client.recreate_collection,
            collection_name=self.collection_name,
            vectors_config={
                "dense": self._models.VectorParams(
                    size=self.vector_size,
                    distance=self._models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": self._models.SparseVectorParams(
                    modifier=self._models.Modifier.IDF,
                ),
            },
        )
        self._ready = True

    async def search(self, request: PassiveRecallRequest) -> PassiveRecallResponse:
        started_at = perf_counter()
        await self.ensure_ready()
        candidate_limit = passive_candidate_limit(request.max_results)
        diagnostics = PassiveRecallDiagnostics() if request.include_diagnostics else None
        embed_started = perf_counter()
        if self.sparse_encoder is not None:
            dense_task = asyncio.create_task(
                self.embedding_service.generate(
                    GenerateEmbeddingsRequest(
                        texts=[request.query],
                        dimensions=self.vector_size,
                        batch_size=1,
                        max_parallel_requests=1,
                        encoding_format=self.dense_encoding_format,
                        normalize=True,
                    )
                )
            )
            sparse_task = asyncio.create_task(self.sparse_encoder.encode([request.query]))
            dense_result, sparse_vectors = await asyncio.gather(dense_task, sparse_task)
            query_sparse = sparse_vectors[0]
        else:
            dense_result = await self.embedding_service.generate(
                GenerateEmbeddingsRequest(
                    texts=[request.query],
                    dimensions=self.vector_size,
                    batch_size=1,
                    max_parallel_requests=1,
                    encoding_format=self.dense_encoding_format,
                    normalize=True,
                )
            )
            query_sparse = None
        embed_ms = (perf_counter() - embed_started) * 1000.0

        query_dense = dense_result.items[0].vector

        must_conditions = [
            self._models.FieldCondition(
                key="status",
                match=self._models.MatchValue(value=RecordStatus.ACTIVE.value),
            )
        ]
        if request.kinds:
            must_conditions.append(
                self._models.FieldCondition(
                    key="type",
                    match=self._models.MatchAny(any=[kind.value for kind in request.kinds]),
                )
            )

        query_started = perf_counter()
        results = await self._run_client_call(
            self.client.query_points,
            collection_name=self.collection_name,
            prefetch=[
                self._models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=candidate_limit,
                    filter=self._models.Filter(must=must_conditions),
                ),
                self._models.Prefetch(
                    query=self._build_sparse_query(
                        request.query,
                        sparse_vector=query_sparse,
                    ),
                    using="sparse",
                    limit=candidate_limit,
                    filter=self._models.Filter(must=must_conditions),
                ),
            ],
            query=self._models.FusionQuery(fusion=self._models.Fusion.RRF),
            limit=candidate_limit,
        )
        query_ms = (perf_counter() - query_started) * 1000.0

        hydration_started = perf_counter()
        ranked = []
        seen_ids: set[str] = set()
        for point in results.points:
            payload = point.payload or {}
            memory_id = str(payload.get("memory_id") or point.id)
            if memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            base_score = float(point.score or 0.0)
            ranked.append(recall_item_from_payload(payload=payload, score=base_score, point_id=point.id))
        hydration_ms = (perf_counter() - hydration_started) * 1000.0

        rerank_started = perf_counter()
        ranked = rerank_passive_items(
            ranked,
            query=request.query,
            include_breakdown=request.include_diagnostics,
        )
        rerank_ms = (perf_counter() - rerank_started) * 1000.0
        select_started = perf_counter()
        selected, total_token_count = select_passive_items(
            ranked,
            max_results=request.max_results,
            token_budget=request.token_budget,
        )
        select_ms = (perf_counter() - select_started) * 1000.0
        total_ms = (perf_counter() - started_at) * 1000.0

        if diagnostics is not None:
            diagnostics.timings_ms = {
                "index_total_ms": round(total_ms, 3),
                "index_embedding_ms": round(embed_ms, 3),
                "index_query_ms": round(query_ms, 3),
                "index_hydration_ms": round(hydration_ms, 3),
                "rerank_ms": round(rerank_ms, 3),
                "selection_ms": round(select_ms, 3),
            }
            diagnostics.counters = {
                "candidate_limit": candidate_limit,
                "candidate_count": len(ranked),
                "selected_count": len(selected),
            }
            diagnostics.flags = {
                "index_used": True,
                "index_fallback_used": False,
                "native_sparse_mode": self.sparse_encoder is None,
                "local_path_index": self._is_local_path,
            }

        return PassiveRecallResponse(
            items=selected,
            total_token_count=total_token_count,
            diagnostics=diagnostics,
        )

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is None:
            return
        result = await self._run_client_call(close)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _import_models():
        from qdrant_client import models

        return models

    def _build_point_vectors(
        self,
        *,
        dense_vector: list[float],
        content: str,
        sparse_vector,
    ) -> dict[str, object]:
        if sparse_vector is not None:
            return {
                "dense": dense_vector,
                "sparse": self._models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
            }

        return {
            "dense": dense_vector,
            "sparse": self._models.Document(text=content, model=self.sparse_model_name),
        }

    def _build_sparse_query(self, query_text: str, *, sparse_vector):
        if sparse_vector is not None:
            return self._models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            )
        return self._models.Document(text=query_text, model=self.sparse_model_name)

    async def _run_client_call(self, func, /, *args, **kwargs):
        if self._is_local_path:
            async with self._client_lock:
                return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)


def _approx_token_count(text: str) -> int:
    return max(len(_tokenize(text)), 1)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _point_id_for_memory_id(memory_id: str) -> str:
    try:
        return str(uuid.UUID(memory_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))


def _supports_qdrant_native_bm25() -> bool:
    raw_version = version("qdrant-client")
    current = _parse_version(raw_version)
    minimum = (1, 15, 2)
    return current >= minimum


def _has_fastembed() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_version(raw_version: str) -> tuple[int, int, int]:
    parts = raw_version.split(".")
    numbers: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits or "0"))
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])
