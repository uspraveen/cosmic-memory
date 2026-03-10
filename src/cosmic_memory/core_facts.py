"""Helpers for first-class core fact handling."""

from __future__ import annotations

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import CoreFactBlock, CoreFactItem, MemoryRecord


def iter_active_core_facts(records: list[MemoryRecord]) -> list[MemoryRecord]:
    core_facts = [
        record
        for record in records
        if record.kind == MemoryKind.CORE_FACT and record.status == RecordStatus.ACTIVE
    ]
    core_facts.sort(
        key=lambda record: (
            int(record.metadata.get("priority", 100)),
            record.updated_at.timestamp(),
        ),
        reverse=True,
    )
    return core_facts


def find_active_core_fact_by_key(
    records: list[MemoryRecord], canonical_key: str
) -> MemoryRecord | None:
    for record in iter_active_core_facts(records):
        if record.metadata.get("canonical_key") == canonical_key:
            return record
    return None


def build_core_fact_block(
    records: list[MemoryRecord],
    *,
    limit: int | None = None,
    max_chars: int = 1500,
) -> CoreFactBlock:
    selected: list[CoreFactItem] = []
    lines: list[str] = []

    for record in iter_active_core_facts(records):
        if not bool(record.metadata.get("always_include", True)):
            continue
        if limit is not None and len(selected) >= limit:
            break

        item = CoreFactItem(
            memory_id=record.memory_id,
            title=record.title,
            content=record.content,
            canonical_key=record.metadata.get("canonical_key"),
            priority=int(record.metadata.get("priority", 100)),
            always_include=bool(record.metadata.get("always_include", True)),
            tags=record.tags,
        )

        line = f"- {item.content}" if not item.title else f"- {item.title}: {item.content}"
        tentative = "\n".join(["# Core Facts", *lines, line]).strip()
        if len(tentative) > max_chars and selected:
            break

        selected.append(item)
        lines.append(line)

    rendered = ""
    if lines:
        rendered = "\n".join(["# Core Facts", *lines])

    return CoreFactBlock(items=selected, rendered=rendered)
