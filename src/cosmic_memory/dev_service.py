"""In-memory development implementation for cosmic-memory."""

from __future__ import annotations

import asyncio
import logging
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
    CoreFactBlock,
    EpisodeIngestResponse,
    GraphStatusResponse,
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
from cosmic_memory.retrieval import (
    apply_graph_recipe_for_mode,
    build_active_response,
    build_active_response_with_graph,
    build_passive_response,
    merge_passive_with_graph,
    passive_candidate_limit,
    search_records,
)

logger = logging.getLogger(__name__)


class InMemoryDevelopmentMemoryService:
    """Small in-memory service used to pin down contracts before real backends exist."""

    def __init__(
        self,
        *,
        graph_store: GraphStore | None = None,
        graph_extractor: GraphExtractionService | None = None,
        graph_llm_extractor: GraphExtractionService | None = None,
        passive_graph_timeout_seconds: float = 0.12,
    ) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self.graph_store = graph_store
        self.graph_extractor = graph_extractor
        self.graph_llm_extractor = graph_llm_extractor
        self.passive_graph_timeout_seconds = passive_graph_timeout_seconds

    async def health(self) -> HealthStatus:
        stats = await self._graph_store_stats()
        return HealthStatus(
            ok=True,
            mode="in_memory_dev",
            graph_enabled=self.graph_store is not None,
            graph_backend=stats.backend if stats is not None else None,
            graph_entity_count=stats.entity_count if stats is not None else 0,
            graph_relation_count=stats.relation_count if stats is not None else 0,
            graph_episode_count=stats.episode_count if stats is not None else 0,
            graph_identity_key_count=stats.identity_key_count if stats is not None else 0,
            graph_extractor_model=getattr(self.graph_extractor, "model_name", None),
            graph_llm_extractor_model=getattr(self.graph_llm_extractor, "model_name", None),
        )

    async def write(self, request: WriteMemoryRequest) -> MemoryRecord:
        record = MemoryRecord(
            kind=request.kind,
            title=request.title,
            content=request.content,
            tags=request.tags,
            metadata=request.metadata,
            provenance=request.provenance,
        )
        await self._prepare_record(record)
        self._records[record.memory_id] = record
        await self._sync_graph_record(record)
        return record

    async def ingest_episode(self, request: IngestEpisodeRequest) -> EpisodeIngestResponse:
        record = await self.write(build_episode_write_request(request))
        return build_episode_ingest_response(
            record,
            observation_count=len(request.observations),
        )

    async def write_core_fact(self, request: WriteCoreFactRequest) -> MemoryRecord:
        if request.canonical_key:
            existing = find_active_core_fact_by_key(
                list(self._records.values()), request.canonical_key
            )
            if existing is not None:
                replacement = await self.supersede(
                    existing.memory_id,
                    SupersedeMemoryRequest(replacement=request.to_write_request()),
                )
                if replacement is not None:
                    return replacement

        return await self.write(request.to_write_request())

    async def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    async def build_core_fact_block(
        self, *, limit: int | None = None, max_chars: int = 1500
    ) -> CoreFactBlock:
        return build_core_fact_block(
            list(self._records.values()),
            limit=limit,
            max_chars=max_chars,
        )

    async def passive_recall(self, request: PassiveRecallRequest) -> PassiveRecallResponse:
        service_started = perf_counter()
        diagnostics = PassiveRecallDiagnostics() if request.include_diagnostics else None
        if diagnostics is not None:
            diagnostics.flags["graph_assist_requested"] = False
            diagnostics.flags["graph_assist_used"] = False
            diagnostics.flags["index_used"] = False
            diagnostics.flags["index_fallback_used"] = False

        base_started = perf_counter()
        base_task = asyncio.create_task(
            asyncio.to_thread(self._build_base_passive_response, request, diagnostics)
        )
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

        response = await base_task
        diagnostics = response.diagnostics or diagnostics
        if diagnostics is not None:
            diagnostics.flags["graph_assist_requested"] = graph_task is not None
            diagnostics.flags.setdefault("graph_assist_used", False)
            diagnostics.timings_ms["service_base_recall_ms"] = round(
                (perf_counter() - base_started) * 1000.0,
                3,
            )
        if graph_task is None:
            return self._finalize_passive_response(
                response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        graph_wait_started = perf_counter()
        try:
            graph_result = await asyncio.wait_for(
                asyncio.shield(graph_task),
                timeout=self.passive_graph_timeout_seconds,
            )
        except asyncio.TimeoutError:
            graph_task.cancel()
            if diagnostics is not None:
                diagnostics.flags["graph_assist_timed_out"] = True
                diagnostics.notes.append("Passive graph assist timed out.")
                diagnostics.timings_ms["graph_wait_ms"] = round(
                    (perf_counter() - graph_wait_started) * 1000.0,
                    3,
                )
            return self._finalize_passive_response(
                response,
                diagnostics=diagnostics,
                started_at=service_started,
            )
        except Exception:
            if diagnostics is not None:
                diagnostics.notes.append("Passive graph assist failed.")
                diagnostics.timings_ms["graph_wait_ms"] = round(
                    (perf_counter() - graph_wait_started) * 1000.0,
                    3,
                )
            return self._finalize_passive_response(
                response,
                diagnostics=diagnostics,
                started_at=service_started,
            )

        if diagnostics is not None:
            diagnostics.timings_ms["graph_wait_ms"] = round(
                (perf_counter() - graph_wait_started) * 1000.0,
                3,
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
        graph_records = [
            self._records[memory_id]
            for memory_id in graph_result.supporting_memory_ids
            if memory_id in self._records and self._records[memory_id].status == RecordStatus.ACTIVE
        ]
        if diagnostics is not None:
            diagnostics.timings_ms["graph_load_records_ms"] = round(
                (perf_counter() - graph_load_started) * 1000.0,
                3,
            )
        if not graph_records:
            if diagnostics is not None:
                diagnostics.notes.append("Graph assist returned no active canonical records.")
            return self._finalize_passive_response(
                response,
                diagnostics=diagnostics,
                started_at=service_started,
            )
        merge_started = perf_counter()
        response = merge_passive_with_graph(
            base_response=response,
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
            graph_records = [
                self._records[memory_id]
                for memory_id in graph_result.supporting_memory_ids
                if memory_id in self._records and self._records[memory_id].status == RecordStatus.ACTIVE
            ]
            if diagnostics is not None:
                diagnostics.timings_ms["graph_load_records_ms"] = round(
                    (perf_counter() - load_started) * 1000.0,
                    3,
                )
            match_started = perf_counter()
            matches = search_records(
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
                diagnostics.counters["matched_memory_count"] = len(matches)
            response = build_active_response_with_graph(
                matches=matches,
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
        matches = search_records(
            self._records.values(),
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
        return IndexStatusResponse(enabled=False, canonical_count=len(self._records))

    async def sync_index(self) -> IndexSyncResponse:
        status = await self.get_index_status()
        return IndexSyncResponse(
            enabled=False,
            mode="sync",
            canonical_count=len(self._records),
            status=status,
        )

    async def rebuild_index(self) -> IndexSyncResponse:
        status = await self.get_index_status()
        return IndexSyncResponse(
            enabled=False,
            mode="rebuild",
            canonical_count=len(self._records),
            status=status,
        )

    async def get_graph_status(self) -> GraphStatusResponse:
        active_records = [record for record in self._records.values() if record.status == RecordStatus.ACTIVE]
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
        )

    async def sync_graph(self) -> GraphSyncResponse:
        return await self._sync_graph(mode="sync")

    async def rebuild_graph(self) -> GraphSyncResponse:
        return await self._sync_graph(mode="rebuild")

    async def supersede(
        self, memory_id: str, request: SupersedeMemoryRequest
    ) -> MemoryRecord | None:
        current = self._records.get(memory_id)
        if current is None:
            return None

        current.status = RecordStatus.SUPERSEDED
        current.updated_at = utc_now()
        await self._sync_graph_record(current)

        replacement = await self.write(request.replacement)
        replacement.supersedes = current.memory_id
        replacement.version = current.version + 1
        current.superseded_by = replacement.memory_id
        current.updated_at = replacement.created_at

        self._records[current.memory_id] = current
        self._records[replacement.memory_id] = replacement
        return replacement

    async def _graph_store_stats(self) -> GraphStoreStats | None:
        if self.graph_store is None:
            return None
        return await self.graph_store.stats()

    async def _sync_graph(self, *, mode: str) -> GraphSyncResponse:
        active_records = [record for record in self._records.values() if record.status == RecordStatus.ACTIVE]
        eligible_count = sum(1 for record in active_records if should_extract_graph_for_record(record))
        persisted_count = sum(
            1 for record in active_records if record.metadata.get("graph_document") is not None
        )
        if self.graph_store is None:
            status = await self.get_graph_status()
            return GraphSyncResponse(
                enabled=False,
                backend=None,
                mode=mode,
                active_memory_count=len(active_records),
                eligible_memory_count=eligible_count,
                persisted_graph_document_count=persisted_count,
                graph_upserts=0,
                graph_removals=0,
                status=status,
            )

        if mode == "rebuild":
            await self.graph_store.reset()

        graph_upserts = 0
        graph_removals = 0
        for record in list(self._records.values()):
            if record.status != RecordStatus.ACTIVE:
                await self.graph_store.remove_memory(record.memory_id)
                graph_removals += 1
                continue
            result = await self._sync_graph_record(record, allow_llm=False)
            if result is not None:
                graph_upserts += 1

        status = await self.get_graph_status()
        return GraphSyncResponse(
            enabled=True,
            backend=status.backend,
            mode=mode,
            active_memory_count=len(active_records),
            eligible_memory_count=eligible_count,
            persisted_graph_document_count=persisted_count,
            graph_upserts=graph_upserts,
            graph_removals=graph_removals,
            status=status,
        )

    async def _sync_graph_record(
        self,
        record: MemoryRecord,
        *,
        allow_llm: bool = False,
    ):
        if self.graph_store is None:
            return None
        await self.graph_store.remove_memory(record.memory_id)
        if record.status != RecordStatus.ACTIVE:
            return None
        document = await ensure_graph_document_for_record(
            record,
            extractor=self.graph_extractor,
            llm_extractor=self.graph_llm_extractor,
            allow_llm=allow_llm,
        )
        if document is None:
            return None
        return await self.graph_store.ingest_document(document)

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

    def _build_base_passive_response(
        self,
        request: PassiveRecallRequest,
        diagnostics: PassiveRecallDiagnostics | None = None,
    ) -> PassiveRecallResponse:
        candidate_limit = passive_candidate_limit(request.max_results)
        search_started = perf_counter()
        matches = search_records(
            self._records.values(),
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
        return build_passive_response(
            matches,
            query=request.query,
            max_results=request.max_results,
            token_budget=request.token_budget,
            include_breakdown=request.include_diagnostics,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _should_run_passive_graph_assist(query_frame) -> bool:
        return bool(query_frame.identity_candidates) or query_frame.prefer_current_state or any(
            intent.value != "generic" for intent in query_frame.intents
        )

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
