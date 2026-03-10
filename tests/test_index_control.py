import asyncio

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import MemoryProvenance, WriteMemoryRequest, utc_now
from cosmic_memory.filesystem_service import FilesystemMemoryService
from tests.tests_support import FakePassiveIndex


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_sync_index_reconciles_registry_and_orphaned_index(tmp_path):
    async def run():
        index = FakePassiveIndex()
        service = FilesystemMemoryService(tmp_path, passive_index=index)
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.SESSION_SUMMARY,
                title="Session",
                content="User worked on Cosmic memory sync and recall.",
                tags=["architecture"],
                provenance=provenance(),
            )
        )

        service.registry.delete_many([record.memory_id])
        index.synced["mem_orphan"] = (RecordStatus.ACTIVE, "orphan.md", "oldhash")

        before = await service.get_index_status()
        assert before.missing_from_registry == [record.memory_id]
        assert before.orphaned_in_index == ["mem_orphan"]

        result = await service.sync_index()

        assert result.registry_upserts == 1
        assert result.indexed_deletes == 1
        assert result.status.missing_from_registry == []
        assert result.status.orphaned_in_index == []
        assert record.memory_id in index.synced
        assert "mem_orphan" not in index.synced

    asyncio.run(run())


def test_sync_index_reindexes_when_canonical_markdown_changes(tmp_path):
    async def run():
        index = FakePassiveIndex()
        service = FilesystemMemoryService(tmp_path, passive_index=index)
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers.",
                tags=["preference"],
                provenance=provenance(),
            )
        )

        updated = record.model_copy(
            update={
                "content": "User prefers concise answers with short bullets.",
                "updated_at": utc_now(),
            }
        )
        service.record_store.write(updated)

        before = await service.get_index_status()
        assert before.stale_registry == [record.memory_id]
        assert before.stale_in_index == [record.memory_id]

        result = await service.sync_index()

        assert result.indexed_upserts == 1
        assert result.status.stale_registry == []
        assert result.status.stale_in_index == []
        refreshed = await service.get(record.memory_id)
        assert refreshed is not None
        assert refreshed.content == "User prefers concise answers with short bullets."

    asyncio.run(run())


def test_rebuild_index_resets_and_rehydrates(tmp_path):
    async def run():
        index = FakePassiveIndex()
        service = FilesystemMemoryService(tmp_path, passive_index=index)
        first = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Profile",
                content="User likes Cosmic.",
                tags=["profile"],
                provenance=provenance(),
            )
        )
        second = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title="Learning",
                content="Agent learned the user prefers concise summaries.",
                tags=["learning"],
                provenance=provenance(),
            )
        )
        index.synced["mem_orphan"] = (RecordStatus.ACTIVE, "orphan.md", "oldhash")

        result = await service.rebuild_index()

        assert result.mode == "rebuild"
        assert result.indexed_upserts == 2
        assert result.indexed_deletes == 3
        assert set(index.synced) == {first.memory_id, second.memory_id}
        assert result.status.orphaned_in_index == []

    asyncio.run(run())
