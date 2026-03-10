"""No-op passive recall backend."""

from collections.abc import Sequence

from cosmic_memory.domain.models import (
    CanonicalMemorySnapshot,
    IndexPointState,
    PassiveRecallRequest,
    PassiveRecallResponse,
)


class NullPassiveMemoryIndex:
    """No-op index implementation used when passive indexing is disabled."""

    async def ensure_ready(self) -> None:
        return None

    async def sync_record(self, snapshot: CanonicalMemorySnapshot) -> None:
        return None

    async def sync_records(self, records: Sequence[CanonicalMemorySnapshot]) -> None:
        return None

    async def snapshot(self) -> dict[str, IndexPointState]:
        return {}

    async def delete_records(self, memory_ids: Sequence[str]) -> None:
        return None

    async def reset(self) -> None:
        return None

    async def search(self, request: PassiveRecallRequest) -> PassiveRecallResponse:
        return PassiveRecallResponse(items=[])

    async def close(self) -> None:
        return None
