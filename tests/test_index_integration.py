import asyncio

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    MemoryProvenance,
    PassiveRecallRequest,
    WriteMemoryRequest,
)
from cosmic_memory.filesystem_service import FilesystemMemoryService
from tests.tests_support import FakePassiveIndex


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_filesystem_service_syncs_passive_index_and_uses_it_for_search(tmp_path):
    async def run():
        index = FakePassiveIndex()
        service = FilesystemMemoryService(tmp_path, passive_index=index)

        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.SESSION_SUMMARY,
                title="Session",
                content="User worked on Cosmic memory architecture.",
                tags=["architecture"],
                provenance=provenance(),
            )
        )

        recall = await service.passive_recall(
            PassiveRecallRequest(
                query="memory architecture",
                max_results=5,
                include_diagnostics=True,
            )
        )

        assert index.synced
        assert record.memory_id in index.synced
        assert index.synced[record.memory_id][0] == RecordStatus.ACTIVE
        assert index.search_requests
        assert recall.items[0].memory_id == "mem_from_index"
        assert recall.diagnostics is not None
        assert recall.diagnostics.timings_ms["service_total_ms"] >= 0
        assert recall.diagnostics.flags["graph_assist_requested"] is False

    asyncio.run(run())
