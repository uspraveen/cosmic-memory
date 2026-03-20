import asyncio

from cosmic_memory.domain.models import MemoryProvenance, WriteCoreFactRequest
from cosmic_memory.filesystem_service import FilesystemMemoryService


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_write_core_fact_supersedes_by_canonical_key(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)

        original = await service.write_core_fact(
            WriteCoreFactRequest(
                title="Name",
                fact="User's name is Praveen.",
                canonical_key="user.name",
                priority=200,
                provenance=provenance(),
            )
        )

        updated = await service.write_core_fact(
            WriteCoreFactRequest(
                title="Name",
                fact="User's name is Praveen Raj.",
                canonical_key="user.name",
                priority=200,
                provenance=provenance(),
            )
        )

        old_record = await service.get(original.memory_id)
        assert old_record is not None
        assert old_record.superseded_by == updated.memory_id
        assert updated.supersedes == original.memory_id

    asyncio.run(run())


def test_core_fact_block_is_deterministic(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)

        await service.write_core_fact(
            WriteCoreFactRequest(
                title="Preference",
                fact="User prefers concise answers.",
                canonical_key="preferences.response_style",
                priority=300,
                provenance=provenance(),
            )
        )
        await service.write_core_fact(
            WriteCoreFactRequest(
                title="Relationship",
                fact="Spouse is Anjana.",
                canonical_key="relationships.spouse",
                priority=250,
                provenance=provenance(),
            )
        )

        block = await service.build_core_fact_block()
        assert len(block.items) == 2
        assert block.items[0].canonical_key == "preferences.response_style"
        assert "# Core Facts" in block.rendered
        assert "User prefers concise answers." in block.rendered

    asyncio.run(run())


def test_core_fact_block_excludes_unconfirmed_relationship_and_contested_facts(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)

        await service.write_core_fact(
            WriteCoreFactRequest(
                title="Preference",
                fact="User prefers concise answers.",
                canonical_key="preferences.response_style",
                priority=300,
                provenance=provenance(),
            )
        )
        await service.write_core_fact(
            WriteCoreFactRequest(
                title="Partner name",
                fact="Partner is Priya.",
                canonical_key="relationships.partner.name",
                priority=250,
                confirmation_status="unconfirmed",
                provenance=provenance(),
            )
        )
        await service.write_core_fact(
            WriteCoreFactRequest(
                title="Favorite color",
                fact="Favorite color is blue.",
                canonical_key="preferences.favorite_color",
                priority=200,
                confirmation_status="contested",
                contested_reason="User said this was incorrect.",
                provenance=provenance(),
            )
        )

        block = await service.build_core_fact_block()

        assert len(block.items) == 1
        assert block.items[0].canonical_key == "preferences.response_style"
        assert "Partner is Priya." not in block.rendered
        assert "Favorite color is blue." not in block.rendered

    asyncio.run(run())
