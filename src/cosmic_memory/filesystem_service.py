"""Filesystem-backed canonical memory service."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from cosmic_memory.core_facts import build_core_fact_block, find_active_core_fact_by_key
from cosmic_memory.domain.enums import RecordStatus
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    ActiveRecallResponse,
    CanonicalMemorySnapshot,
    CoreFactBlock,
    HealthStatus,
    IndexStatusResponse,
    IndexSyncResponse,
    MemoryRecord,
    PassiveRecallDiagnostics,
    PassiveRecallRequest,
    PassiveRecallResponse,
    SupersedeMemoryRequest,
    WriteCoreFactRequest,
    WriteMemoryRequest,
    utc_now,
)
from cosmic_memory.graph import GraphStore, build_query_frame, graph_document_from_memory_record
from cosmic_memory.index.base import PassiveMemoryIndex
from cosmic_memory.retrieval import (
    build_active_response,
    build_active_response_with_graph,
    build_passive_response,
    merge_passive_with_graph,
    passive_candidate_limit,
    search_records,
)
from cosmic_memory.storage import MarkdownRecordStore, SQLiteMemoryRegistry
from cosmic_memory.storage.markdown_store import approx_token_count

logger = logging.getLogger(__name__)


class FilesystemMemoryService:
    """Canonical memory service backed by Markdown records and a SQLite registry."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        passive_index: PassiveMemoryIndex | None = None,
        graph_store: GraphStore | None = None,
        index_sync_batch_size: int = 128,
        passive_graph_timeout_seconds: float = 0.12,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.record_store = MarkdownRecordStore(self.root_dir / "memory")
        self.registry = SQLiteMemoryRegistry(self.root_dir / "registry.db")
        self.passive_index = passive_index
        self.graph_store = graph_store
        self.index_sync_batch_size = index_sync_batch_size
        self.passive_graph_timeout_seconds = passive_graph_timeout_seconds

    async def health(self) -> HealthStatus:
        if self.passive_index is not None:
            await self.passive_index.ensure_ready()
        return HealthStatus(ok=True, mode="filesystem_canonical")

    async def write(self, request: WriteMemoryRequest) -> MemoryRecord:
        record = MemoryRecord(
            kind=request.kind,
            title=request.title,
            content=request.content,
            tags=request.tags,
            metadata=request.metadata,
            provenance=request.provenance,
        )
        await self._persist(record)
        return record

    async def write_core_fact(self, request: WriteCoreFactRequest) -> MemoryRecord:
        write_request = request.to_write_request()
        if request.canonical_key:
            existing = find_active_core_fact_by_key(
                self._load_records(status=RecordStatus.ACTIVE, kinds=[write_request.kind]),
                request.canonical_key,
            )
            if existing is not None:
                replacement = await self.supersede(
                    existing.memory_id,
                    SupersedeMemoryRequest(replacement=write_request),
                )
                if replacement is not None:
                    return replacement

        return await self.write(write_request)

    async def get(self, memory_id: str) -> MemoryRecord | None:
        entry = self.registry.get(memory_id)
        if entry is None:
            return None

        path = Path(entry.path)
        if not path.exists():
            return None

        return self.record_store.read(path)

    async def build_core_fact_block(
        self, *, limit: int | None = None, max_chars: int = 1500
    ) -> CoreFactBlock:
        return build_core_fact_block(
            self._load_records(status=RecordStatus.ACTIVE, kinds=None),
            limit=limit,
            max_chars=max_chars,
        )

    async def passive_recall(self, request: PassiveRecallRequest) -> PassiveRecallResponse:
        service_started = perf_counter()
        diagnostics = PassiveRecallDiagnostics() if request.include_diagnostics else None
        if diagnostics is not None:
            diagnostics.flags["graph_assist_requested"] = False
            diagnostics.flags["graph_assist_used"] = False
            diagnostics.flags["index_fallback_used"] = False

        base_started = perf_counter()
        base_task = asyncio.create_task(self._base_passive_recall(request, diagnostics=diagnostics))
        query_frame = None
        graph_task = None

        if self.graph_store is not None:
            query_frame = build_query_frame(request.query)
            if self._should_run_passive_graph_assist(query_frame):
                if diagnostics is not None:
                    diagnostics.flags["graph_assist_requested"] = True
                graph_task = asyncio.create_task(
                    self.graph_store.passive_search(
                        query_frame,
                        max_entities=max(request.max_results, 3),
                        max_relations=max(request.max_results, 4),
                    )
                )

        base_response = await base_task
        diagnostics = base_response.diagnostics or diagnostics
        if diagnostics is not None:
            diagnostics.flags["graph_assist_requested"] = graph_task is not None
            diagnostics.flags.setdefault("graph_assist_used", False)
            diagnostics.timings_ms["service_base_recall_ms"] = round(
                (perf_counter() - base_started) * 1000.0,
                3,
            )
        if graph_task is None:
            return self._finalize_passive_response(
                base_response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        graph_wait_started = perf_counter()
        graph_result = await self._await_graph_result(
            graph_task,
            started_at=service_started,
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            diagnostics.timings_ms["graph_wait_ms"] = round(
                (perf_counter() - graph_wait_started) * 1000.0,
                3,
            )
        if graph_result is None or not graph_result.supporting_memory_ids:
            return self._finalize_passive_response(
                base_response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        graph_load_started = perf_counter()
        graph_records = self._load_records_by_ids(graph_result.supporting_memory_ids)
        if diagnostics is not None:
            diagnostics.timings_ms["graph_load_records_ms"] = round(
                (perf_counter() - graph_load_started) * 1000.0,
                3,
            )
        if not graph_records:
            if diagnostics is not None:
                diagnostics.notes.append("Graph assist returned no active canonical records.")
            return self._finalize_passive_response(
                base_response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        merge_started = perf_counter()
        response = merge_passive_with_graph(
            base_response=base_response,
            graph_records=graph_records,
            query=request.query,
            kinds=request.kinds,
            max_results=request.max_results,
            token_budget=request.token_budget,
            include_breakdown=request.include_diagnostics,
        )
        diagnostics = response.diagnostics or diagnostics
        if diagnostics is not None:
            diagnostics.timings_ms["graph_merge_ms"] = round(
                (perf_counter() - merge_started) * 1000.0,
                3,
            )
        return self._finalize_passive_response(
            response,
            diagnostics=diagnostics,
            started_at=service_started,
        )

    async def active_recall(self, request: ActiveRecallRequest) -> ActiveRecallResponse:
        if self.graph_store is not None:
            graph_result = await self.graph_store.traverse(
                build_query_frame(request.query),
                seed_entity_ids=request.seed_entities or None,
                max_hops=request.max_hops,
                max_entities=request.max_results,
                max_relations=request.max_results,
            )
            graph_records = self._load_records_by_ids(graph_result.supporting_memory_ids)
            graph_matches = search_records(
                graph_records,
                request.query,
                request.kinds,
                limit=request.max_results,
                score_boosts={memory_id: 0.5 for memory_id in graph_result.supporting_memory_ids},
            )
            return build_active_response_with_graph(
                matches=graph_matches,
                graph_result=graph_result,
            )

        records = self._load_records(status=RecordStatus.ACTIVE, kinds=request.kinds)
        matches = search_records(
            records,
            request.query,
            request.kinds,
            limit=request.max_results,
        )
        return build_active_response(matches)

    async def get_index_status(self) -> IndexStatusResponse:
        snapshots = self._scan_canonical_snapshots()
        return await self._build_index_status(snapshots)

    async def sync_index(self) -> IndexSyncResponse:
        snapshots = self._scan_canonical_snapshots()
        before = await self._build_index_status(snapshots)
        self.registry.replace_all(snapshots)

        indexed_upserts = 0
        indexed_deletes = 0
        if self.passive_index is not None:
            await self.passive_index.ensure_ready()
            to_sync = self._select_snapshots(snapshots, before.missing_from_index, before.stale_in_index)
            await self._sync_snapshot_batches(to_sync)
            indexed_upserts = len(to_sync)
            if before.orphaned_in_index:
                await self.passive_index.delete_records(before.orphaned_in_index)
                indexed_deletes = len(before.orphaned_in_index)

        status = await self._build_index_status(snapshots)
        return IndexSyncResponse(
            enabled=self.passive_index is not None,
            collection_name=self._collection_name(),
            mode="sync",
            canonical_count=len(snapshots),
            registry_upserts=len(snapshots),
            registry_deletes=len(before.orphaned_registry),
            indexed_upserts=indexed_upserts,
            indexed_deletes=indexed_deletes,
            status=status,
        )

    async def rebuild_index(self) -> IndexSyncResponse:
        snapshots = self._scan_canonical_snapshots()
        before = await self._build_index_status(snapshots)
        self.registry.replace_all(snapshots)

        indexed_upserts = 0
        indexed_deletes = 0
        if self.passive_index is not None:
            await self.passive_index.reset()
            await self._sync_snapshot_batches(snapshots)
            indexed_upserts = len(snapshots)
            indexed_deletes = before.indexed_count

        status = await self._build_index_status(snapshots)
        return IndexSyncResponse(
            enabled=self.passive_index is not None,
            collection_name=self._collection_name(),
            mode="rebuild",
            canonical_count=len(snapshots),
            registry_upserts=len(snapshots),
            registry_deletes=len(before.orphaned_registry),
            indexed_upserts=indexed_upserts,
            indexed_deletes=indexed_deletes,
            status=status,
        )

    async def supersede(
        self, memory_id: str, request: SupersedeMemoryRequest
    ) -> MemoryRecord | None:
        current = await self.get(memory_id)
        if current is None:
            return None

        now = utc_now()
        replacement = MemoryRecord(
            kind=request.replacement.kind,
            title=request.replacement.title,
            content=request.replacement.content,
            tags=request.replacement.tags,
            metadata=request.replacement.metadata,
            provenance=request.replacement.provenance,
            version=current.version + 1,
            supersedes=current.memory_id,
            created_at=now,
            updated_at=now,
        )

        current.status = RecordStatus.SUPERSEDED
        current.superseded_by = replacement.memory_id
        current.updated_at = now

        await self._persist(current)
        await self._persist(replacement)
        return replacement

    async def _persist(self, record: MemoryRecord) -> None:
        write_result = self.record_store.write(record)
        self.registry.upsert(record, write_result.path, write_result.content_hash)
        await self._sync_graph_record(record)
        if self.passive_index is not None:
            await self.passive_index.ensure_ready()
            await self.passive_index.sync_record(
                CanonicalMemorySnapshot(
                    memory_id=record.memory_id,
                    kind=record.kind,
                    status=record.status,
                    version=record.version,
                    path=str(write_result.path),
                    content_hash=write_result.content_hash,
                    token_count=approx_token_count(record.content),
                    record=record,
                )
            )

    def _load_records(
        self,
        *,
        status: RecordStatus | None,
        kinds,
    ) -> list[MemoryRecord]:
        entries = self.registry.list(status=status, kinds=kinds)
        records: list[MemoryRecord] = []
        for entry in entries:
            path = Path(entry.path)
            if not path.exists():
                continue
            records.append(self.record_store.read(path))
        return records

    def _load_records_by_ids(self, memory_ids: list[str]) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for memory_id in memory_ids:
            entry = self.registry.get(memory_id)
            if entry is None:
                continue
            path = Path(entry.path)
            if not path.exists():
                continue
            record = self.record_store.read(path)
            if record.status == RecordStatus.ACTIVE:
                records.append(record)
        return records

    def _scan_canonical_snapshots(self) -> list[CanonicalMemorySnapshot]:
        return self.record_store.scan()

    async def _build_index_status(
        self,
        snapshots: list[CanonicalMemorySnapshot],
    ) -> IndexStatusResponse:
        canonical_by_id = {snapshot.memory_id: snapshot for snapshot in snapshots}
        registry_entries = self.registry.list(status=None, kinds=None)
        registry_by_id = {entry.memory_id: entry for entry in registry_entries}

        missing_from_registry = sorted(canonical_by_id.keys() - registry_by_id.keys())
        orphaned_registry = sorted(registry_by_id.keys() - canonical_by_id.keys())
        stale_registry = sorted(
            memory_id
            for memory_id in (canonical_by_id.keys() & registry_by_id.keys())
            if self._registry_entry_is_stale(registry_by_id[memory_id], canonical_by_id[memory_id])
        )

        base_status = IndexStatusResponse(
            enabled=self.passive_index is not None,
            collection_name=self._collection_name(),
            canonical_count=len(snapshots),
            registry_count=len(registry_entries),
            indexed_count=0,
            active_count=sum(1 for snapshot in snapshots if snapshot.status == RecordStatus.ACTIVE),
            superseded_count=sum(
                1 for snapshot in snapshots if snapshot.status == RecordStatus.SUPERSEDED
            ),
            deleted_count=sum(1 for snapshot in snapshots if snapshot.status == RecordStatus.DELETED),
            missing_from_registry=missing_from_registry,
            stale_registry=stale_registry,
            orphaned_registry=orphaned_registry,
        )

        if self.passive_index is None:
            return base_status

        await self.passive_index.ensure_ready()
        index_snapshot = await self.passive_index.snapshot()
        base_status.indexed_count = len(index_snapshot)
        base_status.missing_from_index = sorted(canonical_by_id.keys() - index_snapshot.keys())
        base_status.orphaned_in_index = sorted(index_snapshot.keys() - canonical_by_id.keys())
        base_status.stale_in_index = sorted(
            memory_id
            for memory_id in (canonical_by_id.keys() & index_snapshot.keys())
            if self._index_entry_is_stale(index_snapshot[memory_id], canonical_by_id[memory_id])
        )
        return base_status

    async def _sync_snapshot_batches(self, snapshots: Iterable[CanonicalMemorySnapshot]) -> None:
        batch: list[CanonicalMemorySnapshot] = []
        for snapshot in snapshots:
            batch.append(snapshot)
            if len(batch) >= self.index_sync_batch_size:
                await self.passive_index.sync_records(batch)
                batch = []
        if batch:
            await self.passive_index.sync_records(batch)

    @staticmethod
    def _select_snapshots(
        snapshots: list[CanonicalMemorySnapshot],
        missing_ids: list[str],
        stale_ids: list[str],
    ) -> list[CanonicalMemorySnapshot]:
        target_ids = set(missing_ids) | set(stale_ids)
        return [snapshot for snapshot in snapshots if snapshot.memory_id in target_ids]

    @staticmethod
    def _registry_entry_is_stale(entry, snapshot: CanonicalMemorySnapshot) -> bool:
        return (
            entry.content_hash != snapshot.content_hash
            or Path(entry.path) != Path(snapshot.path)
            or entry.status != snapshot.status
            or entry.kind != snapshot.kind
            or entry.version != snapshot.version
        )

    @staticmethod
    def _index_entry_is_stale(entry, snapshot: CanonicalMemorySnapshot) -> bool:
        return entry.content_hash != snapshot.content_hash or entry.status != snapshot.status

    def _collection_name(self) -> str | None:
        return getattr(self.passive_index, "collection_name", None)

    async def _base_passive_recall(
        self,
        request: PassiveRecallRequest,
        *,
        diagnostics: PassiveRecallDiagnostics | None = None,
    ) -> PassiveRecallResponse:
        if self.passive_index is not None:
            try:
                await self.passive_index.ensure_ready()
                return await self.passive_index.search(request)
            except Exception:
                logger.exception("Passive index search failed, falling back to lexical recall.")
                if diagnostics is not None:
                    diagnostics.flags["index_fallback_used"] = True
                    diagnostics.notes.append(
                        "Passive index search failed; fell back to lexical recall."
                    )

        search_started = perf_counter()
        records = self._load_records(status=RecordStatus.ACTIVE, kinds=request.kinds)
        candidate_limit = passive_candidate_limit(request.max_results)
        matches = search_records(
            records,
            request.query,
            request.kinds,
            limit=candidate_limit,
        )
        if diagnostics is not None:
            diagnostics.timings_ms["lexical_search_ms"] = round(
                (perf_counter() - search_started) * 1000.0,
                3,
            )
            diagnostics.counters["candidate_limit"] = candidate_limit
            diagnostics.flags.setdefault("index_used", False)
        return build_passive_response(
            matches,
            query=request.query,
            max_results=request.max_results,
            token_budget=request.token_budget,
            include_breakdown=request.include_diagnostics,
            diagnostics=diagnostics,
        )

    async def _await_graph_result(
        self,
        task: asyncio.Task,
        *,
        started_at: float,
        diagnostics: PassiveRecallDiagnostics | None = None,
    ):
        if task.done():
            return self._task_result_or_none(task)

        remaining = self.passive_graph_timeout_seconds - (perf_counter() - started_at)
        if remaining <= 0:
            task.cancel()
            if diagnostics is not None:
                diagnostics.flags["graph_assist_timed_out"] = True
                diagnostics.notes.append("Passive graph assist skipped because no timeout budget remained.")
            return None

        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.TimeoutError:
            task.cancel()
            if diagnostics is not None:
                diagnostics.flags["graph_assist_timed_out"] = True
                diagnostics.notes.append("Passive graph assist timed out.")
            return None
        except Exception:
            logger.exception("Passive graph assist failed.")
            if diagnostics is not None:
                diagnostics.notes.append("Passive graph assist failed.")
            return None

    @staticmethod
    def _should_run_passive_graph_assist(query_frame) -> bool:
        return bool(query_frame.identity_candidates) or query_frame.prefer_current_state or any(
            intent.value != "generic" for intent in query_frame.intents
        )

    @staticmethod
    def _task_result_or_none(task: asyncio.Task):
        try:
            return task.result()
        except asyncio.CancelledError:
            return None
        except Exception:
            logger.exception("Passive graph assist failed.")
            return None

    async def _sync_graph_record(self, record: MemoryRecord) -> None:
        if self.graph_store is None:
            return
        await self.graph_store.remove_memory(record.memory_id)
        if record.status != RecordStatus.ACTIVE:
            return
        document = graph_document_from_memory_record(record)
        if document is None:
            return
        await self.graph_store.ingest_document(document)

    @staticmethod
    def _finalize_passive_response(
        response: PassiveRecallResponse,
        *,
        diagnostics: PassiveRecallDiagnostics | None,
        started_at: float,
    ) -> PassiveRecallResponse:
        if diagnostics is None:
            return response

        diagnostics.timings_ms["service_total_ms"] = round(
            (perf_counter() - started_at) * 1000.0,
            3,
        )
        if response.diagnostics is diagnostics:
            return response
        return response.model_copy(update={"diagnostics": diagnostics})
