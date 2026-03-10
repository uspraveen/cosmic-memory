"""Abstract service contract for cosmic-memory."""

from __future__ import annotations

from typing import Protocol

from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    ActiveRecallResponse,
    CoreFactBlock,
    HealthStatus,
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

    async def write_core_fact(self, request: WriteCoreFactRequest) -> MemoryRecord: ...

    async def get(self, memory_id: str) -> MemoryRecord | None: ...

    async def build_core_fact_block(
        self, *, limit: int | None = None, max_chars: int = 1500
    ) -> CoreFactBlock: ...

    async def passive_recall(self, request: PassiveRecallRequest) -> PassiveRecallResponse: ...

    async def active_recall(self, request: ActiveRecallRequest) -> ActiveRecallResponse: ...

    async def get_index_status(self) -> IndexStatusResponse: ...

    async def sync_index(self) -> IndexSyncResponse: ...

    async def rebuild_index(self) -> IndexSyncResponse: ...

    async def supersede(
        self, memory_id: str, request: SupersedeMemoryRequest
    ) -> MemoryRecord | None: ...
