"""Qdrant-backed entity similarity index for graph candidate generation."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Sequence

from cosmic_memory.domain.models import GenerateEmbeddingsRequest
from cosmic_memory.embeddings.base import EmbeddingService
from cosmic_memory.graph.entity_index import (
    EntitySimilarityHit,
    entity_similarity_text,
)
from cosmic_memory.graph.models import GraphEntityNode
from cosmic_memory.graph.ontology import EntityType


class QdrantEntitySimilarityIndex:
    """Dense entity-level similarity index backed by Qdrant."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        collection_name: str = "memory_entities",
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
                "qdrant-client is required for QdrantEntitySimilarityIndex. "
                "Install project dependencies with `python -m pip install -e .`."
            ) from exc

        self._models = self._import_models()
        self.client = QdrantClient(url=url, path=path)
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self.vector_size = vector_size or embedding_service.dimensions
        self.embed_batch_size = embed_batch_size
        self.embed_parallel_requests = embed_parallel_requests
        self.dense_encoding_format = dense_encoding_format
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        exists = False
        if hasattr(self.client, "collection_exists"):
            exists = await asyncio.to_thread(self.client.collection_exists, self.collection_name)
        if not exists:
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection_name,
                vectors_config=self._models.VectorParams(
                    size=self.vector_size,
                    distance=self._models.Distance.COSINE,
                ),
            )
        self._ready = True

    async def sync_entity(self, entity: GraphEntityNode) -> None:
        await self.sync_entities([entity])

    async def sync_entities(self, entities: Sequence[GraphEntityNode]) -> None:
        if not entities:
            return
        await self.ensure_ready()
        result = await self.embedding_service.generate(
            GenerateEmbeddingsRequest(
                texts=[entity_similarity_text(entity) for entity in entities],
                dimensions=self.vector_size,
                batch_size=min(self.embed_batch_size, max(len(entities), 1)),
                max_parallel_requests=self.embed_parallel_requests,
                encoding_format=self.dense_encoding_format,
                normalize=True,
            )
        )
        vectors = [item.vector for item in sorted(result.items, key=lambda item: item.index)]
        points = []
        for entity, vector in zip(entities, vectors, strict=True):
            points.append(
                self._models.PointStruct(
                    id=_point_id_for_entity_id(entity.entity_id),
                    vector=vector,
                    payload={
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type.value,
                        "canonical_name": entity.canonical_name,
                        "alias_values": entity.alias_values,
                        "memory_ids": entity.memory_ids,
                        "resolution_state": entity.resolution_state,
                        "updated_at": entity.updated_at.isoformat(),
                        "search_text": entity_similarity_text(entity),
                    },
                )
            )
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points,
        )

    async def delete_entities(self, entity_ids: Sequence[str]) -> None:
        if not entity_ids:
            return
        await self.ensure_ready()
        point_ids = [_point_id_for_entity_id(entity_id) for entity_id in entity_ids]
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=point_ids,
            wait=True,
        )

    async def search(
        self,
        query: str,
        *,
        entity_types: Sequence[EntityType] | None = None,
        limit: int = 8,
    ) -> list[EntitySimilarityHit]:
        if not query.strip():
            return []
        await self.ensure_ready()
        result = await self.embedding_service.generate(
            GenerateEmbeddingsRequest(
                texts=[query],
                dimensions=self.vector_size,
                batch_size=1,
                max_parallel_requests=1,
                encoding_format=self.dense_encoding_format,
                normalize=True,
            )
        )
        must_conditions = []
        if entity_types:
            must_conditions.append(
                self._models.FieldCondition(
                    key="entity_type",
                    match=self._models.MatchAny(any=[entity_type.value for entity_type in entity_types]),
                )
            )
        points = await asyncio.to_thread(
            self.client.query_points,
            collection_name=self.collection_name,
            query=result.items[0].vector,
            limit=limit,
            query_filter=self._models.Filter(must=must_conditions) if must_conditions else None,
        )
        hits: list[EntitySimilarityHit] = []
        for point in points.points:
            payload = point.payload or {}
            entity_type = payload.get("entity_type")
            if entity_type is None:
                continue
            hits.append(
                EntitySimilarityHit(
                    entity_id=payload.get("entity_id") or str(point.id),
                    score=float(point.score or 0.0),
                    entity_type=EntityType(entity_type),
                    canonical_name=payload.get("canonical_name") or payload.get("entity_id") or str(point.id),
                    alias_values=list(payload.get("alias_values") or []),
                    memory_ids=list(payload.get("memory_ids") or []),
                    resolution_state=payload.get("resolution_state") or "canonical",
                )
            )
        return hits

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _import_models():
        from qdrant_client import models

        return models


def _point_id_for_entity_id(entity_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cosmic-memory-entity::{entity_id}"))
