"""Helpers for first-class core fact handling."""

from __future__ import annotations

from cosmic_memory.domain.enums import CoreFactConfirmationStatus, MemoryKind, RecordStatus
from cosmic_memory.domain.models import CoreFactBlock, CoreFactItem, MemoryRecord

_RELATIONSHIP_OR_IDENTITY_KEY_PREFIXES = (
    "relationships.",
    "identity.",
    "user.identity.",
    "profile.identity.",
)


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


def _confirmation_status(record: MemoryRecord) -> CoreFactConfirmationStatus:
    raw_status = record.metadata.get("confirmation_status")
    if isinstance(raw_status, str):
        try:
            return CoreFactConfirmationStatus(raw_status.strip().lower())
        except ValueError:
            pass
    return CoreFactConfirmationStatus.CONFIRMED


def _requires_confirmed_core_fact(record: MemoryRecord) -> bool:
    canonical_key = str(record.metadata.get("canonical_key") or "").strip().lower()
    if any(canonical_key.startswith(prefix) for prefix in _RELATIONSHIP_OR_IDENTITY_KEY_PREFIXES):
        return True
    tags = [str(tag or "").strip().lower() for tag in record.tags]
    return "relationship" in tags or "identity" in tags


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
        confirmation_status = _confirmation_status(record)
        if confirmation_status == CoreFactConfirmationStatus.CONTESTED:
            continue
        if _requires_confirmed_core_fact(record) and confirmation_status != CoreFactConfirmationStatus.CONFIRMED:
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
            confirmation_status=confirmation_status,
            source_type=record.provenance.source_kind if record.provenance else None,
            source_id=record.provenance.source_id if record.provenance else None,
            created_in_session_id=record.metadata.get("created_in_session_id")
            or (record.provenance.session_id if record.provenance else None),
            created_by_tool=record.metadata.get("created_by_tool"),
            derived_from_assistant_inference=bool(
                record.metadata.get("derived_from_assistant_inference", False)
            ),
            contested_at=record.metadata.get("contested_at"),
            contested_reason=record.metadata.get("contested_reason"),
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
