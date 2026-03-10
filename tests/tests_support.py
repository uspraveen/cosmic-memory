from collections.abc import Sequence

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    CanonicalMemorySnapshot,
    IndexPointState,
    PassiveRecallRequest,
    PassiveRecallResponse,
    RecallItem,
)


class FakePassiveIndex:
    def __init__(self) -> None:
        self.synced: dict[str, tuple[RecordStatus, str, str | None]] = {}
        self.search_requests: list[PassiveRecallRequest] = []

    async def ensure_ready(self) -> None:
        return None

    async def sync_record(self, snapshot: CanonicalMemorySnapshot) -> None:
        self.synced[snapshot.memory_id] = (
            snapshot.status,
            snapshot.path,
            snapshot.content_hash,
        )

    async def sync_records(self, records: Sequence[CanonicalMemorySnapshot]) -> None:
        for snapshot in records:
            await self.sync_record(snapshot)

    async def snapshot(self) -> dict[str, IndexPointState]:
        return {
            memory_id: IndexPointState(
                memory_id=memory_id,
                status=status,
                content_hash=content_hash,
            )
            for memory_id, (status, _path, content_hash) in self.synced.items()
        }

    async def delete_records(self, memory_ids: Sequence[str]) -> None:
        for memory_id in memory_ids:
            self.synced.pop(memory_id, None)

    async def reset(self) -> None:
        self.synced = {}

    async def search(self, request: PassiveRecallRequest) -> PassiveRecallResponse:
        self.search_requests.append(request)
        return PassiveRecallResponse(
            items=[
                RecallItem(
                    memory_id="mem_from_index",
                    kind=MemoryKind.AGENT_NOTE,
                    title="Indexed note",
                    content="Indexed recall result.",
                    score=0.99,
                    tags=["indexed"],
                    token_count=3,
                )
            ],
            total_token_count=3,
        )

    async def close(self) -> None:
        return None
