"""Base passive recall index contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cosmic_memory.domain.models import (
    CanonicalMemorySnapshot,
    IndexPointState,
    PassiveRecallRequest,
    PassiveRecallResponse,
)


class PassiveMemoryIndex(Protocol):
    """Passive recall backend contract."""

    async def ensure_ready(self) -> None: ...

    async def sync_record(self, snapshot: CanonicalMemorySnapshot) -> None: ...

    async def sync_records(self, records: Sequence[CanonicalMemorySnapshot]) -> None: ...

    async def snapshot(self) -> dict[str, IndexPointState]: ...

    async def delete_records(self, memory_ids: Sequence[str]) -> None: ...

    async def reset(self) -> None: ...

    async def search(self, request: PassiveRecallRequest) -> PassiveRecallResponse: ...

    async def close(self) -> None: ...
