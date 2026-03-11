"""Filesystem layout helpers for canonical memory records."""

import re
from pathlib import Path

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryRecord


KIND_DIRECTORY_MAP = {
    MemoryKind.CORE_FACT: "core_facts",
    MemoryKind.SESSION_SUMMARY: "sessions",
    MemoryKind.TASK_SUMMARY: "tasks",
    MemoryKind.AGENT_NOTE: "agent_notes",
    MemoryKind.USER_DATA: "user_data",
    MemoryKind.TRANSCRIPT: "transcripts",
}


def directory_for_kind(memory_root: Path, kind: MemoryKind) -> Path:
    return memory_root / KIND_DIRECTORY_MAP[kind]


def path_for_record(memory_root: Path, record: MemoryRecord) -> Path:
    directory = directory_for_kind(memory_root, record.kind)
    filename = f"{record.memory_id}.md"

    if record.kind is MemoryKind.SESSION_SUMMARY:
        session_id = _record_key(record, "session_id")
        if session_id:
            filename = f"{session_id}.md"
    elif record.kind is MemoryKind.TASK_SUMMARY:
        task_id = _record_key(record, "task_id")
        if task_id:
            filename = f"{task_id}.md"
    elif record.kind is MemoryKind.AGENT_NOTE:
        agent_id = _record_key(record, "agent_id") or _agent_id_from_provenance(record)
        if agent_id:
            directory = directory / _safe_path_component(agent_id)
            filename = "learnings.md"

    return directory / filename


def _record_key(record: MemoryRecord, key: str) -> str | None:
    raw = record.metadata.get(key)
    if raw is None:
        raw = getattr(record.provenance, key, None)
    value = str(raw or "").strip()
    return value or None


def _agent_id_from_provenance(record: MemoryRecord) -> str | None:
    created_by = str(record.provenance.created_by or "").strip()
    return created_by or None


def _safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "unknown_agent"
