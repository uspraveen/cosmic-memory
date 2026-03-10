"""Filesystem layout helpers for canonical memory records."""

from pathlib import Path

from cosmic_memory.domain.enums import MemoryKind


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


def path_for_record(memory_root: Path, memory_id: str, kind: MemoryKind) -> Path:
    return directory_for_kind(memory_root, kind) / f"{memory_id}.md"
