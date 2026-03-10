from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, MemoryRecord
from cosmic_memory.storage.registry import SQLiteMemoryRegistry


def test_registry_upsert_and_get(tmp_path):
    registry = SQLiteMemoryRegistry(tmp_path / "registry.db")
    record = MemoryRecord(
        kind=MemoryKind.CORE_FACT,
        title="Profile fact",
        content="User prefers concise answers.",
        tags=["preference"],
        provenance=MemoryProvenance(source_kind="gateway", created_by="test"),
    )

    registry.upsert(record, tmp_path / "memory" / "core_facts" / f"{record.memory_id}.md", "abc123")
    entry = registry.get(record.memory_id)

    assert entry is not None
    assert entry.memory_id == record.memory_id
    assert entry.kind == MemoryKind.CORE_FACT
    assert entry.content_hash == "abc123"
    assert entry.tags == ["preference"]
