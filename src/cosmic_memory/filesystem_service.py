"""Filesystem-backed canonical memory service."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from cosmic_memory.control_surface import (
    CurrentStateRequest,
    CurrentStateResponse,
    MemoryBriefRequest,
    MemoryBriefResponse,
    MemoryQueryPlanRequest,
    MemoryQueryPlanResponse,
    ResolveIdentityCandidate,
    ResolveIdentityRequest,
    ResolveIdentityResponse,
    ResolvedEntityRef,
    SchemaContextResponse,
    TemporalFactsRequest,
    TemporalFactsResponse,
    build_brief_findings,
    build_memory_query_plan,
    build_schema_context,
    filter_current_state_facts,
    filter_temporal_facts,
    graph_facts_from_active_response,
)
from cosmic_memory.core_facts import build_core_fact_block, find_active_core_fact_by_key
from cosmic_memory.domain.enums import RecordStatus
from cosmic_memory.domain.models import (
    ActiveRecallDiagnostics,
    ActiveRecallRequest,
    ActiveRecallResponse,
    CanonicalMemorySnapshot,
    CoreFactBlock,
    EpisodeIngestResponse,
    HealthStatus,
    IngestEpisodeRequest,
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
from cosmic_memory.episode_ingestion import (
    build_episode_ingest_response,
    build_episode_write_request,
)
from cosmic_memory.extraction.base import GraphExtractionService
from cosmic_memory.graph import GraphStore, build_query_frame, ensure_graph_document_for_record
from cosmic_memory.index.base import PassiveMemoryIndex
from cosmic_memory.retrieval import (
    apply_graph_recipe_for_mode,
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
        graph_extractor: GraphExtractionService | None = None,
        index_sync_batch_size: int = 128,
        passive_graph_timeout_seconds: float = 0.12,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.record_store = MarkdownRecordStore(self.root_dir / "memory")
        self.registry = SQLiteMemoryRegistry(self.root_dir / "registry.db")
        self.passive_index = passive_index
        self.graph_store = graph_store
        self.graph_extractor = graph_extractor
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

    async def ingest_episode(self, request: IngestEpisodeRequest) -> EpisodeIngestResponse:
        record = await self.write(build_episode_write_request(request))
        return build_episode_ingest_response(
            record,
            observation_count=len(request.observations),
        )

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

        recipe_started = perf_counter()
        recipe_application = apply_graph_recipe_for_mode(
            graph_result=graph_result,
            query_frame=query_frame,
            mode="passive",
            max_results=request.max_results,
        )
        graph_result = recipe_application.graph_result
        if diagnostics is not None:
            diagnostics.timings_ms["graph_recipe_ms"] = round(
                (perf_counter() - recipe_started) * 1000.0,
                3,
            )
            diagnostics.notes.append(f"graph recipe: {recipe_application.recipe_name}")

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
            graph_memory_boosts=recipe_application.memory_boosts,
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
        service_started = perf_counter()
        diagnostics = ActiveRecallDiagnostics() if request.include_diagnostics else None
        if self.graph_store is not None:
            if diagnostics is not None:
                diagnostics.flags["graph_used"] = True
            query_frame = build_query_frame(request.query)
            graph_started = perf_counter()
            graph_result = await self.graph_store.traverse(
                query_frame,
                seed_entity_ids=request.seed_entities or None,
                max_hops=request.max_hops,
                max_entities=request.max_results,
                max_relations=request.max_results,
            )
            if diagnostics is not None:
                diagnostics.timings_ms["graph_traverse_ms"] = round(
                    (perf_counter() - graph_started) * 1000.0,
                    3,
                )
            recipe_started = perf_counter()
            recipe_application = apply_graph_recipe_for_mode(
                graph_result=graph_result,
                query_frame=query_frame,
                mode="active",
                max_results=request.max_results,
            )
            graph_result = recipe_application.graph_result
            if diagnostics is not None:
                diagnostics.timings_ms["graph_recipe_ms"] = round(
                    (perf_counter() - recipe_started) * 1000.0,
                    3,
                )
                diagnostics.notes.append(f"graph recipe: {recipe_application.recipe_name}")
            load_started = perf_counter()
            graph_records = self._load_records_by_ids(graph_result.supporting_memory_ids)
            if diagnostics is not None:
                diagnostics.timings_ms["graph_load_records_ms"] = round(
                    (perf_counter() - load_started) * 1000.0,
                    3,
                )
            match_started = perf_counter()
            graph_matches = search_records(
                graph_records,
                request.query,
                request.kinds,
                limit=request.max_results,
                score_boosts=recipe_application.memory_boosts
                or {memory_id: 0.5 for memory_id in graph_result.supporting_memory_ids},
            )
            if diagnostics is not None:
                diagnostics.timings_ms["graph_match_ms"] = round(
                    (perf_counter() - match_started) * 1000.0,
                    3,
                )
                diagnostics.counters["supporting_memory_count"] = len(graph_result.supporting_memory_ids)
                diagnostics.counters["entity_count"] = len(graph_result.entities)
                diagnostics.counters["relation_count"] = len(graph_result.relations)
                diagnostics.counters["matched_memory_count"] = len(graph_matches)
            response = build_active_response_with_graph(
                matches=graph_matches,
                graph_result=graph_result,
                diagnostics=diagnostics,
            )
            return self._finalize_active_response(
                response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        if diagnostics is not None:
            diagnostics.flags["graph_used"] = False
        search_started = perf_counter()
        records = self._load_records(status=RecordStatus.ACTIVE, kinds=request.kinds)
        matches = search_records(
            records,
            request.query,
            request.kinds,
            limit=request.max_results,
        )
        if diagnostics is not None:
            diagnostics.timings_ms["lexical_search_ms"] = round(
                (perf_counter() - search_started) * 1000.0,
                3,
            )
            diagnostics.counters["matched_memory_count"] = len(matches)
        response = build_active_response(matches, diagnostics=diagnostics)
        return self._finalize_active_response(
            response,
            diagnostics=diagnostics,
            started_at=service_started,
        )

    async def get_schema_context(self) -> SchemaContextResponse:
        return build_schema_context(graph_available=self.graph_store is not None)

    async def plan_query(self, request: MemoryQueryPlanRequest) -> MemoryQueryPlanResponse:
        return build_memory_query_plan(request, graph_available=self.graph_store is not None)

    async def resolve_identity(self, request: ResolveIdentityRequest) -> ResolveIdentityResponse:
        from cosmic_memory.graph import GraphIdentityCandidate, IdentityKeyType, build_identity_key

        candidate = GraphIdentityCandidate(
            key_type=IdentityKeyType(request.key_type),
            raw_value=request.value,
            provider=request.provider,
            verified=request.verified,
            confidence=request.confidence,
        )
        key = build_identity_key(candidate)
        if self.graph_store is None:
            return ResolveIdentityResponse(
                graph_available=False,
                status="no_match",
                normalized_key_id=key.key_id,
                normalized_value=key.normalized_value,
            )

        resolution = await self.graph_store.resolve_identity(candidate)
        entity = None
        if resolution.entity_id is not None:
            resolved = await self.graph_store.get_entity(resolution.entity_id)
            if resolved is not None:
                entity = ResolvedEntityRef(
                    entity_id=resolved.entity_id,
                    name=resolved.canonical_name,
                    entity_type=resolved.entity_type.value,
                    memory_ids=resolved.memory_ids,
                )
        candidates: list[ResolveIdentityCandidate] = []
        for item in resolution.candidates:
            resolved = await self.graph_store.get_entity(item.entity_id)
            candidates.append(
                ResolveIdentityCandidate(
                    entity_id=item.entity_id,
                    reason=item.reason,
                    confidence=item.confidence,
                    name=resolved.canonical_name if resolved is not None else None,
                    entity_type=resolved.entity_type.value if resolved is not None else None,
                )
            )
        return ResolveIdentityResponse(
            graph_available=True,
            status=resolution.status,
            normalized_key_id=key.key_id,
            normalized_value=key.normalized_value,
            entity=entity,
            candidates=candidates,
        )

    async def get_current_state(self, request: CurrentStateRequest) -> CurrentStateResponse:
        query = request.query
        if "current" not in query.lower() and "active" not in query.lower() and "now" not in query.lower():
            query = f"{query} current active now"
        active = await self.active_recall(
            ActiveRecallRequest(
                query=query,
                max_results=request.max_results,
                max_hops=request.max_hops,
                include_diagnostics=request.include_diagnostics,
            )
        )
        facts = filter_current_state_facts(graph_facts_from_active_response(active))
        supporting_ids = {memory_id for fact in facts for memory_id in fact.memory_ids}
        supporting_memories = (
            [item for item in active.items if not supporting_ids or item.memory_id in supporting_ids]
            if request.include_supporting_memories
            else []
        )
        return CurrentStateResponse(
            query=request.query,
            facts=facts[: request.max_results],
            entities=active.entities,
            supporting_memories=supporting_memories[: request.max_results],
            search_plan=[*active.search_plan, "filter relations to current-state facts"],
            diagnostics=active.diagnostics.model_dump(mode="json") if active.diagnostics else None,
        )

    async def get_temporal_facts(self, request: TemporalFactsRequest) -> TemporalFactsResponse:
        query = request.query
        temporal_terms = {"before", "after", "when", "last", "history", "timeline", "during"}
        if not any(term in query.lower() for term in temporal_terms):
            query = f"{query} before after when last timeline"
        active = await self.active_recall(
            ActiveRecallRequest(
                query=query,
                max_results=request.max_results,
                max_hops=request.max_hops,
                include_diagnostics=request.include_diagnostics,
            )
        )
        facts = filter_temporal_facts(graph_facts_from_active_response(active))
        supporting_ids = {memory_id for fact in facts for memory_id in fact.memory_ids}
        supporting_memories = (
            [item for item in active.items if not supporting_ids or item.memory_id in supporting_ids]
            if request.include_supporting_memories
            else []
        )
        return TemporalFactsResponse(
            query=request.query,
            facts=facts[: request.max_results],
            entities=active.entities,
            supporting_memories=supporting_memories[: request.max_results],
            search_plan=[*active.search_plan, "filter relations to temporal facts"],
            diagnostics=active.diagnostics.model_dump(mode="json") if active.diagnostics else None,
        )

    async def build_memory_brief(self, request: MemoryBriefRequest) -> MemoryBriefResponse:
        plan = build_memory_query_plan(
            MemoryQueryPlanRequest(query=request.query, max_hops=request.max_hops),
            graph_available=self.graph_store is not None,
        )
        passive_task = asyncio.create_task(
            self.passive_recall(
                PassiveRecallRequest(
                    query=request.query,
                    max_results=request.passive_max_results,
                    token_budget=request.token_budget,
                    include_diagnostics=request.include_diagnostics,
                )
            )
        )
        active_task = None
        current_task = None
        temporal_task = None
        core_fact_task = None
        if plan.recommended_mode in {"active", "hybrid"}:
            active_task = asyncio.create_task(
                self.active_recall(
                    ActiveRecallRequest(
                        query=request.query,
                        max_results=request.active_max_results,
                        max_hops=plan.suggested_max_hops,
                        include_diagnostics=request.include_diagnostics,
                    )
                )
            )
        if any(tool == "current_state" for tool in plan.tool_sequence):
            current_task = asyncio.create_task(
                self.get_current_state(
                    CurrentStateRequest(
                        query=request.query,
                        max_results=request.active_max_results,
                        max_hops=plan.suggested_max_hops,
                        include_supporting_memories=True,
                        include_diagnostics=request.include_diagnostics,
                    )
                )
            )
        if any(tool == "temporal_facts" for tool in plan.tool_sequence):
            temporal_task = asyncio.create_task(
                self.get_temporal_facts(
                    TemporalFactsRequest(
                        query=request.query,
                        max_results=request.active_max_results,
                        max_hops=plan.suggested_max_hops,
                        include_supporting_memories=True,
                        include_diagnostics=request.include_diagnostics,
                    )
                )
            )
        if request.include_core_facts:
            core_fact_task = asyncio.create_task(self.build_core_fact_block(limit=8, max_chars=2_500))

        passive = await passive_task
        active = await active_task if active_task is not None else None
        current_state = await current_task if current_task is not None else None
        temporal = await temporal_task if temporal_task is not None else None
        core_facts = await core_fact_task if core_fact_task is not None else None
        findings = build_brief_findings(
            current_state=current_state,
            temporal=temporal,
            active=active,
            passive=passive,
        )
        supporting_memory_ids = sorted(
            {
                *(item.memory_id for item in passive.items),
                *((item.memory_id for item in active.items) if active is not None else ()),
                *((memory_id for fact in current_state.facts for memory_id in fact.memory_ids) if current_state else ()),
                *((memory_id for fact in temporal.facts for memory_id in fact.memory_ids) if temporal else ()),
            }
        )
        return MemoryBriefResponse(
            plan=plan,
            core_facts=core_facts,
            passive=passive,
            active=active,
            current_state=current_state,
            temporal=temporal,
            findings=findings,
            supporting_memory_ids=supporting_memory_ids,
        )

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
        await self._prepare_record(record)
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
        document = await ensure_graph_document_for_record(
            record,
            extractor=self.graph_extractor,
        )
        if document is None:
            return
        await self.graph_store.ingest_document(document)

    async def _prepare_record(self, record: MemoryRecord) -> None:
        try:
            await ensure_graph_document_for_record(
                record,
                extractor=self.graph_extractor,
            )
        except Exception:
            logger.exception("Graph extraction failed for memory %s", record.memory_id)

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

    @staticmethod
    def _finalize_active_response(
        response: ActiveRecallResponse,
        *,
        diagnostics: ActiveRecallDiagnostics | None,
        started_at: float,
    ) -> ActiveRecallResponse:
        if diagnostics is None:
            return response

        diagnostics.timings_ms["service_total_ms"] = round(
            (perf_counter() - started_at) * 1000.0,
            3,
        )
        if response.diagnostics is diagnostics:
            return response
        return response.model_copy(update={"diagnostics": diagnostics})
