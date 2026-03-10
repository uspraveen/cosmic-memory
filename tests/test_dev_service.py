from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    MemoryProvenance,
    PassiveRecallRequest,
    SupersedeMemoryRequest,
    WriteMemoryRequest,
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


async def _seed(service: InMemoryDevelopmentMemoryService):
    return await service.write(
        WriteMemoryRequest(
            kind=MemoryKind.CORE_FACT,
            title="User preference",
            content="User prefers concise answers and wants Cosmic to remember defaults.",
            tags=["preference", "style"],
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


def test_passive_recall_finds_record():
    import asyncio

    async def run():
        service = InMemoryDevelopmentMemoryService()
        await _seed(service)
        result = await service.passive_recall(
            PassiveRecallRequest(query="concise answers", max_results=5)
        )
        assert len(result.items) == 1
        assert result.items[0].kind == MemoryKind.CORE_FACT

    asyncio.run(run())


def test_active_recall_returns_entities():
    import asyncio

    async def run():
        service = InMemoryDevelopmentMemoryService()
        await _seed(service)
        result = await service.active_recall(
            ActiveRecallRequest(query="user preferences", max_results=5)
        )
        assert len(result.items) == 1
        assert len(result.entities) == 1
        assert result.entities[0].entity_id == "user"

    asyncio.run(run())


def test_supersede_marks_old_record():
    import asyncio

    async def run():
        service = InMemoryDevelopmentMemoryService()
        original = await _seed(service)
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

        current = await service.get(original.memory_id)
        assert replacement is not None
        assert current is not None
        assert current.status == RecordStatus.SUPERSEDED
        assert current.superseded_by == replacement.memory_id
        assert replacement.supersedes == current.memory_id

    asyncio.run(run())
