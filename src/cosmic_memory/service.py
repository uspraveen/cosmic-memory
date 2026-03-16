"""Abstract service contract for cosmic-memory."""

from __future__ import annotations

from typing import Protocol

from cosmic_memory.control_surface import (
    CurrentStateRequest,
    CurrentStateResponse,
    MemoryBriefRequest,
    MemoryBriefResponse,
    MemoryQueryPlanRequest,
    MemoryQueryPlanResponse,
    ResolveIdentityRequest,
    ResolveIdentityResponse,
    SchemaContextResponse,
    TemporalFactsRequest,
    TemporalFactsResponse,
)
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    ActiveRecallResponse,
    CoreFactBlock,
    EpisodeIngestResponse,
    GraphStatusResponse,
    GraphSyncRequest,
    GraphSyncResponse,
    HealthStatus,
    IngestEpisodeRequest,
    IndexStatusResponse,
    IndexSyncResponse,
    MemoryRecord,
    PassiveRecallRequest,
    PassiveRecallResponse,
    SupersedeMemoryRequest,
    WriteCoreFactRequest,
    WriteMemoryRequest,
)


class MemoryService(Protocol):
    """Service contract for long-term memory operations."""

    async def health(self) -> HealthStatus: ...

    async def write(self, request: WriteMemoryRequest) -> MemoryRecord: ...

    async def ingest_episode(self, request: IngestEpisodeRequest) -> EpisodeIngestResponse: ...

    async def write_core_fact(self, request: WriteCoreFactRequest) -> MemoryRecord: ...

    async def get(self, memory_id: str) -> MemoryRecord | None: ...

    async def build_core_fact_block(
        self, *, limit: int | None = None, max_chars: int = 1500
    ) -> CoreFactBlock: ...

    async def passive_recall(self, request: PassiveRecallRequest) -> PassiveRecallResponse: ...

    async def active_recall(self, request: ActiveRecallRequest) -> ActiveRecallResponse: ...

    async def get_schema_context(self) -> SchemaContextResponse: ...

    async def plan_query(self, request: MemoryQueryPlanRequest) -> MemoryQueryPlanResponse: ...

    async def resolve_identity(self, request: ResolveIdentityRequest) -> ResolveIdentityResponse: ...

    async def get_current_state(self, request: CurrentStateRequest) -> CurrentStateResponse: ...

    async def get_temporal_facts(self, request: TemporalFactsRequest) -> TemporalFactsResponse: ...

    async def build_memory_brief(self, request: MemoryBriefRequest) -> MemoryBriefResponse: ...

    async def get_index_status(self) -> IndexStatusResponse: ...

    async def sync_index(self) -> IndexSyncResponse: ...

    async def rebuild_index(self) -> IndexSyncResponse: ...

    async def get_graph_status(self) -> GraphStatusResponse: ...

    async def sync_graph(self, request: GraphSyncRequest | None = None) -> GraphSyncResponse: ...

    async def rebuild_graph(self, request: GraphSyncRequest | None = None) -> GraphSyncResponse: ...

    async def supersede(
        self, memory_id: str, request: SupersedeMemoryRequest
    ) -> MemoryRecord | None: ...
