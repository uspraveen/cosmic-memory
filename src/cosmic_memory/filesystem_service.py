"""Filesystem-backed canonical memory service."""

from __future__ import annotations

import asyncio
import contextlib
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
    GraphStatusResponse,
    GraphSyncRequest,
    GraphStoreStats,
    GraphSyncResponse,
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
from cosmic_memory.graph import (
    GraphStore,
    build_query_frame,
    ensure_graph_document_for_record,
    should_extract_graph_for_record,
)
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
from cosmic_memory.storage.markdown_store import approx_token_count, canonical_record_hash

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
        graph_llm_extractor: GraphExtractionService | None = None,
        index_sync_batch_size: int = 128,
        passive_graph_timeout_seconds: float = 0.12,
        async_graph_writes: bool = False,
        graph_write_worker_poll_seconds: float = 0.5,
        graph_write_retry_base_seconds: float = 5.0,
        graph_write_retry_max_seconds: float = 300.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.record_store = MarkdownRecordStore(self.root_dir / "memory")
        self.registry = SQLiteMemoryRegistry(self.root_dir / "registry.db")
        self.passive_index = passive_index
        self.graph_store = graph_store
        self.graph_extractor = graph_extractor
        self.graph_llm_extractor = graph_llm_extractor
        self.index_sync_batch_size = index_sync_batch_size
        self.passive_graph_timeout_seconds = passive_graph_timeout_seconds
        self.async_graph_writes = async_graph_writes
        self.graph_write_worker_poll_seconds = max(0.05, graph_write_worker_poll_seconds)
        self.graph_write_retry_base_seconds = max(0.1, graph_write_retry_base_seconds)
        self.graph_write_retry_max_seconds = max(
            self.graph_write_retry_base_seconds,
            graph_write_retry_max_seconds,
        )
        self._graph_write_event = asyncio.Event()
        self._graph_write_shutdown = asyncio.Event()
        self._graph_write_task: asyncio.Task | None = None
        self._graph_operation_lock = asyncio.Lock()

    async def health(self) -> HealthStatus:
        if self.passive_index is not None:
            await self.passive_index.ensure_ready()
        stats = await self._graph_store_stats()
        queue_counts = self.registry.graph_sync_queue_counts()
        return HealthStatus(
            ok=True,
            mode="filesystem_canonical",
            graph_enabled=self.graph_store is not None,
            graph_backend=stats.backend if stats is not None else None,
            graph_entity_count=stats.entity_count if stats is not None else 0,
            graph_relation_count=stats.relation_count if stats is not None else 0,
            graph_episode_count=stats.episode_count if stats is not None else 0,
            graph_identity_key_count=stats.identity_key_count if stats is not None else 0,
            graph_extractor_model=getattr(self.graph_extractor, "model_name", None),
            graph_llm_extractor_model=getattr(self.graph_llm_extractor, "model_name", None),
            graph_cache_ready=stats.cache_ready if stats is not None else False,
            graph_cache_memory_count=stats.cache_memory_count if stats is not None else 0,
            graph_cache_entity_count=stats.cache_entity_count if stats is not None else 0,
            graph_cache_relation_count=stats.cache_relation_count if stats is not None else 0,
            graph_cache_episode_count=stats.cache_episode_count if stats is not None else 0,
            graph_cache_hydrated_at=stats.cache_hydrated_at if stats is not None else None,
            graph_cache_build_ms=stats.cache_build_ms if stats is not None else None,
            graph_queue_pending_count=queue_counts.get("pending", 0),
            graph_queue_running_count=queue_counts.get("running", 0),
            graph_queue_failed_count=queue_counts.get("failed", 0),
        )

    async def start_background_tasks(self) -> None:
        if not self._graph_writes_enabled():
            return
        requeued = self.registry.requeue_running_graph_sync_jobs()
        if requeued:
            logger.warning("memory.graph_write_jobs_requeued count=%s", requeued)
        self._graph_write_shutdown.clear()
        self._graph_write_event.set()
        if self._graph_write_task is not None and not self._graph_write_task.done():
            return
        self._graph_write_task = asyncio.create_task(
            self._graph_write_worker(),
            name="cosmic-memory-graph-write-worker",
        )

    async def stop_background_tasks(self) -> None:
        self._graph_write_shutdown.set()
        self._graph_write_event.set()
        if self._graph_write_task is None:
            return
        self._graph_write_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._graph_write_task
        self._graph_write_task = None
        self._graph_write_event.clear()
        self._graph_write_shutdown.clear()

    async def wait_for_graph_queue_idle(self, *, timeout_seconds: float = 10.0) -> dict[str, int]:
        deadline = asyncio.get_running_loop().time() + max(0.1, timeout_seconds)
        while True:
            counts = self.registry.graph_sync_queue_counts()
            if counts.get("pending", 0) == 0 and counts.get("running", 0) == 0:
                return counts
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Timed out waiting for graph sync queue to become idle.")
            await asyncio.sleep(0.05)

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

    async def warm_graph_cache(self) -> GraphStoreStats | None:
        if self.graph_store is None:
            return None
        warm_cache = getattr(self.graph_store, "warm_cache", None)
        if warm_cache is None:
            return await self._graph_store_stats()
        logger.info("memory.graph_cache_warm_start")
        result = await warm_cache()
        stats = result if isinstance(result, GraphStoreStats) else await self._graph_store_stats()
        if stats is not None:
            logger.info(
                "memory.graph_cache_warm_complete backend=%s cache_ready=%s cache_entities=%s cache_relations=%s cache_build_ms=%s",
                stats.backend,
                stats.cache_ready,
                stats.cache_entity_count,
                stats.cache_relation_count,
                stats.cache_build_ms,
            )
        return stats

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

    async def get_graph_status(self) -> GraphStatusResponse:
        active_records = self._load_records(status=RecordStatus.ACTIVE, kinds=None)
        stats = await self._graph_store_stats()
        return GraphStatusResponse(
            enabled=self.graph_store is not None,
            backend=stats.backend if stats is not None else None,
            extractor_enabled=self.graph_extractor is not None,
            extractor_model=getattr(self.graph_extractor, "model_name", None),
            llm_extractor_enabled=self.graph_llm_extractor is not None,
            llm_extractor_model=getattr(self.graph_llm_extractor, "model_name", None),
            active_memory_count=len(active_records),
            eligible_memory_count=sum(
                1 for record in active_records if should_extract_graph_for_record(record)
            ),
            persisted_graph_document_count=sum(
                1 for record in active_records if record.metadata.get("graph_document") is not None
            ),
            ingested_memory_count=stats.memory_count if stats is not None else 0,
            entity_count=stats.entity_count if stats is not None else 0,
            relation_count=stats.relation_count if stats is not None else 0,
            episode_count=stats.episode_count if stats is not None else 0,
            identity_key_count=stats.identity_key_count if stats is not None else 0,
            cache_ready=stats.cache_ready if stats is not None else False,
            cache_memory_count=stats.cache_memory_count if stats is not None else 0,
            cache_entity_count=stats.cache_entity_count if stats is not None else 0,
            cache_relation_count=stats.cache_relation_count if stats is not None else 0,
            cache_episode_count=stats.cache_episode_count if stats is not None else 0,
            cache_hydrated_at=stats.cache_hydrated_at if stats is not None else None,
            cache_build_ms=stats.cache_build_ms if stats is not None else None,
        )

    async def sync_graph(self, request: GraphSyncRequest | None = None) -> GraphSyncResponse:
        async with self._graph_operation_lock:
            return await self._sync_graph(mode="sync", request=request or GraphSyncRequest())

    async def rebuild_graph(self, request: GraphSyncRequest | None = None) -> GraphSyncResponse:
        async with self._graph_operation_lock:
            return await self._sync_graph(mode="rebuild", request=request or GraphSyncRequest())

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
        if record.status == RecordStatus.ACTIVE:
            await self._prepare_record(record, allow_llm=not self._graph_writes_enabled())
        snapshot = await self._write_canonical_record(record, sync_passive_index=True)
        if self._graph_writes_enabled():
            allow_llm = record.status == RecordStatus.ACTIVE
            persist_graph_document = record.status == RecordStatus.ACTIVE
            job = self._enqueue_graph_sync(
                memory_id=record.memory_id,
                content_hash=snapshot.content_hash,
                allow_llm=allow_llm,
                persist_graph_document=persist_graph_document,
            )
            logger.info(
                "memory.graph_write_enqueued memory_id=%s kind=%s job_id=%s allow_llm=%s persist_graph_document=%s",
                record.memory_id,
                record.kind.value,
                job.job_id,
                allow_llm,
                persist_graph_document,
            )
            return
        await self._sync_graph_record(record)

    async def _write_canonical_record(
        self,
        record: MemoryRecord,
        *,
        sync_passive_index: bool,
    ) -> CanonicalMemorySnapshot:
        write_result = self.record_store.write(record)
        self.registry.upsert(record, write_result.path, write_result.content_hash)
        snapshot = CanonicalMemorySnapshot(
            memory_id=record.memory_id,
            kind=record.kind,
            status=record.status,
            version=record.version,
            path=str(write_result.path),
            content_hash=write_result.content_hash,
            token_count=approx_token_count(record.content),
            record=record,
        )
        if sync_passive_index and self.passive_index is not None:
            await self.passive_index.ensure_ready()
            await self.passive_index.sync_record(snapshot)
        return snapshot

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

    async def _graph_store_stats(self) -> GraphStoreStats | None:
        if self.graph_store is None:
            return None
        return await self.graph_store.stats()

    async def _sync_graph(
        self,
        *,
        mode: str,
        request: GraphSyncRequest,
    ) -> GraphSyncResponse:
        records = self._load_records(status=None, kinds=None)
        active_records = [record for record in records if record.status == RecordStatus.ACTIVE]
        eligible_count = sum(1 for record in active_records if should_extract_graph_for_record(record))
        persisted_count = sum(
            1 for record in active_records if record.metadata.get("graph_document") is not None
        )
        target_active_records = (
            active_records
            if mode == "rebuild"
            else self._select_graph_target_records(active_records, request=request)
        )
        if self.graph_store is None:
            status = await self.get_graph_status()
            return GraphSyncResponse(
                enabled=False,
                backend=None,
                mode=mode,
                active_memory_count=len(active_records),
                eligible_memory_count=eligible_count,
                target_memory_count=len(target_active_records),
                persisted_graph_document_count=persisted_count,
                persisted_graph_document_writes=0,
                graph_upserts=0,
                graph_removals=0,
                failed_memory_count=0,
                failed_memory_ids=[],
                llm_backfill_enabled=request.allow_llm,
                cache_warmed=False,
                status=status,
            )

        if mode == "rebuild":
            logger.info(
                "memory.graph_rebuild_start active_records=%s allow_llm=%s persist_graph_documents=%s",
                len(active_records),
                request.allow_llm,
                request.persist_graph_documents,
            )
            await self.graph_store.reset()
        else:
            logger.info(
                "memory.graph_sync_start active_records=%s target_records=%s allow_llm=%s persist_graph_documents=%s only_missing_graph_documents=%s max_records=%s requested_memory_ids=%s",
                len(active_records),
                len(target_active_records),
                request.allow_llm,
                request.persist_graph_documents,
                request.only_missing_graph_documents,
                request.max_records,
                len(request.memory_ids),
            )

        graph_upserts = 0
        graph_removals = 0
        persisted_graph_document_writes = 0
        failed_memory_ids: list[str] = []
        target_ids = {record.memory_id for record in target_active_records}
        for record in records:
            if record.status != RecordStatus.ACTIVE:
                try:
                    await self.graph_store.remove_memory(record.memory_id)
                    graph_removals += 1
                except Exception:
                    failed_memory_ids.append(record.memory_id)
                    logger.exception(
                        "memory.graph_record_sync_failed memory_id=%s kind=%s title=%s mode=%s action=remove",
                        record.memory_id,
                        record.kind.value,
                        record.title,
                        mode,
                    )
                continue
            if record.memory_id not in target_ids:
                continue
            try:
                result, metadata_persisted = await self._sync_graph_record(
                    record,
                    allow_llm=request.allow_llm,
                    persist_graph_document=request.persist_graph_documents,
                )
            except Exception:
                failed_memory_ids.append(record.memory_id)
                logger.exception(
                    "memory.graph_record_sync_failed memory_id=%s kind=%s title=%s mode=%s allow_llm=%s persist_graph_document=%s",
                    record.memory_id,
                    record.kind.value,
                    record.title,
                    mode,
                    request.allow_llm,
                    request.persist_graph_documents,
                )
                continue
            if result is not None:
                graph_upserts += 1
            if metadata_persisted:
                persisted_graph_document_writes += 1

        cache_warmed = False
        if self._should_warm_graph_cache(request):
            cache_warmed = await self._warm_graph_cache_after_sync(mode=mode)

        status = await self.get_graph_status()
        logger.info(
            "memory.graph_%s_complete upserts=%s removals=%s failures=%s persisted_graph_document_writes=%s entities=%s relations=%s episodes=%s cache_ready=%s cache_build_ms=%s",
            mode,
            graph_upserts,
            graph_removals,
            len(failed_memory_ids),
            persisted_graph_document_writes,
            status.entity_count,
            status.relation_count,
            status.episode_count,
            status.cache_ready,
            status.cache_build_ms,
        )
        return GraphSyncResponse(
            enabled=True,
            backend=status.backend,
            mode=mode,
            active_memory_count=len(active_records),
            eligible_memory_count=eligible_count,
            target_memory_count=len(target_active_records),
            persisted_graph_document_count=status.persisted_graph_document_count,
            persisted_graph_document_writes=persisted_graph_document_writes,
            graph_upserts=graph_upserts,
            graph_removals=graph_removals,
            failed_memory_count=len(failed_memory_ids),
            failed_memory_ids=failed_memory_ids,
            llm_backfill_enabled=request.allow_llm,
            cache_warmed=cache_warmed,
            status=status,
        )

    def _graph_writes_enabled(self) -> bool:
        return self.async_graph_writes and self.graph_store is not None

    def _enqueue_graph_sync(
        self,
        *,
        memory_id: str,
        content_hash: str,
        allow_llm: bool,
        persist_graph_document: bool,
    ):
        job = self.registry.enqueue_graph_sync(
            memory_id=memory_id,
            content_hash=content_hash,
            allow_llm=allow_llm,
            persist_graph_document=persist_graph_document,
        )
        self._graph_write_event.set()
        return job

    async def _graph_write_worker(self) -> None:
        logger.info("memory.graph_write_worker_started")
        try:
            while not self._graph_write_shutdown.is_set():
                job = self.registry.lease_next_graph_sync_job()
                if job is None:
                    self._graph_write_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._graph_write_event.wait(),
                            timeout=self.graph_write_worker_poll_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue
                await self._process_graph_sync_job(job)
        except asyncio.CancelledError:
            logger.info("memory.graph_write_worker_cancelled")
            raise
        finally:
            logger.info("memory.graph_write_worker_stopped")

    async def _process_graph_sync_job(self, job) -> None:
        lease_token = job.lease_token or ""
        entry = self.registry.get(job.memory_id)
        if entry is None:
            self.registry.mark_graph_sync_job_succeeded(
                job_id=job.job_id,
                lease_token=lease_token,
                status="stale",
                last_error="registry entry missing",
            )
            logger.info(
                "memory.graph_write_job_stale job_id=%s memory_id=%s reason=registry_missing",
                job.job_id,
                job.memory_id,
            )
            return
        if entry.content_hash != job.content_hash:
            self.registry.mark_graph_sync_job_succeeded(
                job_id=job.job_id,
                lease_token=lease_token,
                status="stale",
                last_error="content hash changed",
            )
            logger.info(
                "memory.graph_write_job_stale job_id=%s memory_id=%s reason=content_hash_changed current_hash=%s job_hash=%s",
                job.job_id,
                job.memory_id,
                entry.content_hash,
                job.content_hash,
            )
            return
        path = Path(entry.path)
        if not path.exists():
            self.registry.mark_graph_sync_job_succeeded(
                job_id=job.job_id,
                lease_token=lease_token,
                status="stale",
                last_error="canonical file missing",
            )
            logger.info(
                "memory.graph_write_job_stale job_id=%s memory_id=%s reason=canonical_missing",
                job.job_id,
                job.memory_id,
            )
            return
        try:
            record = self.record_store.read(path)
            async with self._graph_operation_lock:
                await self._sync_graph_record(
                    record,
                    allow_llm=job.allow_llm,
                    persist_graph_document=job.persist_graph_document,
                )
        except Exception as exc:
            retry_delay_seconds = self._graph_retry_delay(job.attempts)
            error_message = f"{type(exc).__name__}: {exc}"
            self.registry.mark_graph_sync_job_failed(
                job_id=job.job_id,
                lease_token=lease_token,
                error_message=error_message[:2000],
                retry_delay_seconds=retry_delay_seconds,
            )
            logger.exception(
                "memory.graph_write_job_failed job_id=%s memory_id=%s attempts=%s retry_delay_seconds=%s",
                job.job_id,
                job.memory_id,
                job.attempts,
                retry_delay_seconds,
            )
            return
        self.registry.mark_graph_sync_job_succeeded(
            job_id=job.job_id,
            lease_token=lease_token,
            status="succeeded",
        )
        logger.info(
            "memory.graph_write_job_succeeded job_id=%s memory_id=%s attempts=%s",
            job.job_id,
            job.memory_id,
            job.attempts,
        )

    def _graph_retry_delay(self, attempts: int) -> float:
        exponent = max(0, attempts - 1)
        return min(
            self.graph_write_retry_max_seconds,
            self.graph_write_retry_base_seconds * (2**exponent),
        )

    async def _sync_graph_record(
        self,
        record: MemoryRecord,
        *,
        allow_llm: bool = False,
        persist_graph_document: bool = False,
    ) -> tuple[object | None, bool]:
        if self.graph_store is None:
            return None, False
        await self.graph_store.remove_memory(record.memory_id)
        if record.status != RecordStatus.ACTIVE:
            logger.info(
                "memory.graph_memory_removed memory_id=%s status=%s",
                record.memory_id,
                record.status.value,
            )
            return None, False
        before_hash = canonical_record_hash(record) if persist_graph_document else None
        document = await ensure_graph_document_for_record(
            record,
            extractor=self.graph_extractor,
            llm_extractor=self.graph_llm_extractor,
            allow_llm=allow_llm,
        )
        if document is None:
            logger.info(
                "memory.graph_ingest_skipped memory_id=%s kind=%s",
                record.memory_id,
                record.kind.value,
            )
            return None, False
        metadata_persisted = False
        if before_hash is not None and canonical_record_hash(record) != before_hash:
            await self._write_canonical_record(record, sync_passive_index=True)
            metadata_persisted = True
            logger.info(
                "memory.graph_document_persisted memory_id=%s kind=%s",
                record.memory_id,
                record.kind.value,
            )
        result = await self.graph_store.ingest_document(document)
        logger.info(
            "memory.graph_ingested memory_id=%s episode_id=%s entities=%s relations=%s invalidated=%s",
            record.memory_id,
            result.episode_id,
            len(result.entity_ids),
            len(result.relation_ids),
            len(result.invalidated_relation_ids),
        )
        return result, metadata_persisted

    def _select_graph_target_records(
        self,
        active_records: list[MemoryRecord],
        *,
        request: GraphSyncRequest,
    ) -> list[MemoryRecord]:
        selected = active_records
        if request.memory_ids:
            target_ids = {memory_id for memory_id in request.memory_ids if memory_id}
            selected = [record for record in selected if record.memory_id in target_ids]
        if request.only_missing_graph_documents:
            selected = [
                record
                for record in selected
                if record.metadata.get("graph_document") is None
            ]
        if request.max_records is not None:
            selected = selected[: request.max_records]
        return selected

    def _should_warm_graph_cache(self, request: GraphSyncRequest) -> bool:
        if self.graph_store is None:
            return False
        warm_cache = getattr(self.graph_store, "warm_cache", None)
        if warm_cache is None:
            return False
        if request.warm_cache is not None:
            return request.warm_cache
        return True

    async def _warm_graph_cache_after_sync(self, *, mode: str) -> bool:
        try:
            stats = await self.warm_graph_cache()
        except Exception:
            logger.exception("memory.graph_cache_warm_failed mode=%s", mode)
            return False
        return bool(stats is not None and stats.cache_ready)

    async def _prepare_record(
        self,
        record: MemoryRecord,
        *,
        allow_llm: bool = True,
    ) -> None:
        try:
            await ensure_graph_document_for_record(
                record,
                extractor=self.graph_extractor,
                llm_extractor=self.graph_llm_extractor,
                allow_llm=allow_llm,
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
