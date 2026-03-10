"""In-memory development implementation for cosmic-memory."""

from __future__ import annotations

import asyncio
from time import perf_counter

from cosmic_memory.core_facts import build_core_fact_block, find_active_core_fact_by_key
from cosmic_memory.domain.enums import RecordStatus
from cosmic_memory.domain.models import (
    ActiveRecallDiagnostics,
    ActiveRecallRequest,
    ActiveRecallResponse,
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
from cosmic_memory.retrieval import (
    build_active_response,
    build_active_response_with_graph,
    build_passive_response,
    merge_passive_with_graph,
    passive_candidate_limit,
    search_records,
)


class InMemoryDevelopmentMemoryService:
    """Small in-memory service used to pin down contracts before real backends exist."""

    def __init__(
        self,
        *,
        graph_store: GraphStore | None = None,
        passive_graph_timeout_seconds: float = 0.12,
    ) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self.graph_store = graph_store
        self.passive_graph_timeout_seconds = passive_graph_timeout_seconds

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, mode="in_memory_dev")

    async def write(self, request: WriteMemoryRequest) -> MemoryRecord:
        record = MemoryRecord(
            kind=request.kind,
            title=request.title,
            content=request.content,
            tags=request.tags,
            metadata=request.metadata,
            provenance=request.provenance,
        )
        self._records[record.memory_id] = record
        await self._sync_graph_record(record)
        return record

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
            graph_started = perf_counter()
            graph_result = await self.graph_store.traverse(
                build_query_frame(request.query),
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
                score_boosts={memory_id: 0.5 for memory_id in graph_result.supporting_memory_ids},
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
