from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, MemoryRecord
from cosmic_memory.storage.markdown_store import MarkdownRecordStore


def test_markdown_roundtrip(tmp_path):
    store = MarkdownRecordStore(tmp_path / "memory")
    record = MemoryRecord(
        kind=MemoryKind.CORE_FACT,
        title="Profile fact",
        content="User prefers concise answers.",
        tags=["preference"],
        provenance=MemoryProvenance(source_kind="gateway", created_by="test"),
    )

    result = store.write(record)
    loaded = store.read(result.path)

    assert loaded.memory_id == record.memory_id
    assert loaded.kind == MemoryKind.CORE_FACT
    assert loaded.content == "User prefers concise answers."
    assert loaded.title == "Profile fact"
