import asyncio

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    MemoryProvenance,
    PassiveRecallRequest,
    SupersedeMemoryRequest,
    WriteMemoryRequest,
)
from cosmic_memory.filesystem_service import FilesystemMemoryService


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_filesystem_service_write_get_and_recall(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers and direct explanations.",
                tags=["preference"],
                provenance=provenance(),
                metadata={
                    "entities": [
                        {
                            "entity_id": "user",
                            "name": "User",
                            "entity_type": "person",
                        }
                    ]
                },
            )
        )

        fetched = await service.get(record.memory_id)
        recall = await service.passive_recall(
            PassiveRecallRequest(query="concise answers", max_results=5)
        )

        assert fetched is not None
        assert fetched.memory_id == record.memory_id
        assert len(recall.items) == 1
        assert recall.items[0].memory_id == record.memory_id

    asyncio.run(run())


def test_filesystem_service_supersede(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        original = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers.",
                tags=["preference"],
                provenance=provenance(),
            )
        )

        replacement = await service.supersede(
            original.memory_id,
            SupersedeMemoryRequest(
                replacement=WriteMemoryRequest(
                    kind=MemoryKind.CORE_FACT,
                    title="Updated preference",
                    content="User now prefers concise answers with sparse bullets.",
                    tags=["preference"],
                    provenance=provenance(),
                )
            ),
        )

        old_record = await service.get(original.memory_id)
        assert replacement is not None
        assert old_record is not None
        assert old_record.status == RecordStatus.SUPERSEDED
        assert old_record.superseded_by == replacement.memory_id
        assert replacement.supersedes == original.memory_id
        assert (tmp_path / "memory" / "core_facts" / f"{replacement.memory_id}.md").exists()

    asyncio.run(run())
