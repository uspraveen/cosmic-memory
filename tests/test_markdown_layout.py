from pathlib import Path

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, MemoryRecord
from cosmic_memory.storage.markdown_store import MarkdownRecordStore


def _record(kind: MemoryKind, *, metadata: dict | None = None, created_by: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        kind=kind,
        title="Example",
        content="Test content",
        metadata=metadata or {},
        provenance=MemoryProvenance(
            source_kind="test",
            source_id="src_123",
            created_by=created_by,
            session_id=(metadata or {}).get("session_id"),
            task_id=(metadata or {}).get("task_id"),
        ),
    )


def test_session_summary_uses_session_id_path(tmp_path: Path) -> None:
    store = MarkdownRecordStore(tmp_path / "memory")
    record = _record(MemoryKind.SESSION_SUMMARY, metadata={"session_id": "sess_20260311"})

    result = store.write(record)

    assert result.path == tmp_path / "memory" / "sessions" / "sess_20260311.md"
    assert result.path.exists()


def test_task_summary_uses_task_id_path(tmp_path: Path) -> None:
    store = MarkdownRecordStore(tmp_path / "memory")
    record = _record(MemoryKind.TASK_SUMMARY, metadata={"task_id": "tsk_abc123"})

    result = store.write(record)

    assert result.path == tmp_path / "memory" / "tasks" / "tsk_abc123.md"
    assert result.path.exists()


def test_agent_note_uses_nested_learnings_path_and_recursive_scan(tmp_path: Path) -> None:
    store = MarkdownRecordStore(tmp_path / "memory")
    record = _record(MemoryKind.AGENT_NOTE, created_by="cosmic/docs-agent:2.1.0")

    result = store.write(record)
    snapshots = store.scan()

    assert result.path == tmp_path / "memory" / "agent_notes" / "cosmic_docs-agent_2.1.0" / "learnings.md"
    assert result.path.exists()
    assert len(snapshots) == 1
    assert snapshots[0].memory_id == record.memory_id
    assert snapshots[0].path == str(result.path)
