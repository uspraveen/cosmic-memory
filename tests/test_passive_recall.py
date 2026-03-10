import asyncio

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, PassiveRecallRequest, WriteMemoryRequest
from cosmic_memory.filesystem_service import FilesystemMemoryService


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_passive_recall_respects_token_budget(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Primary preference",
                content=" ".join(["concise"] * 12),
                tags=["preference"],
                provenance=provenance(),
            )
        )
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Imported document",
                content=" ".join(["concise"] * 40),
                tags=["document"],
                provenance=provenance(),
            )
        )

        recall = await service.passive_recall(
            PassiveRecallRequest(query="concise", max_results=5, token_budget=20)
        )

        assert len(recall.items) == 1
        assert recall.total_token_count <= 20

    asyncio.run(run())


def test_passive_recall_prioritizes_agent_notes_over_user_data(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Imported preference doc",
                content="User prefers concise answers for Cosmic responses.",
                tags=["imported"],
                provenance=provenance(),
            )
        )
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title="Agent learning",
                content="User prefers concise answers for Cosmic responses.",
                tags=["learning"],
                provenance=provenance(),
            )
        )

        recall = await service.passive_recall(
            PassiveRecallRequest(query="concise answers cosmic responses", max_results=5)
        )

        assert len(recall.items) >= 2
        assert recall.items[0].kind == MemoryKind.AGENT_NOTE

    asyncio.run(run())
