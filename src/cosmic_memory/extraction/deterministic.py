"""Deterministic graph extraction for obvious personal-memory writes."""

from __future__ import annotations

import re

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryRecord
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.graph.models import GraphIdentityCandidate
from cosmic_memory.graph.ontology import EntityType, IdentityKeyType, RelationType

_PRIMARY_USER_FALLBACK_NAME = "Primary User"
_SUPPORTED_KINDS = {
    MemoryKind.CORE_FACT,
    MemoryKind.USER_DATA,
    MemoryKind.AGENT_NOTE,
    MemoryKind.SESSION_SUMMARY,
    MemoryKind.TASK_SUMMARY,
}
_FIRST_PERSON_PREFIX = r"(?:i|i'm|i am|my|mine|me|user|the user)"
_NAME_PATTERNS = (
    re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})?\s*name\s+is\s+(?P<value>[A-Za-z][A-Za-z .'-]{{1,79}})\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:i am|i'm)\s+(?P<value>[A-Z][A-Za-z .'-]{1,79})\s*\.?\s*$"),
)
_RELATION_PATTERNS: tuple[tuple[RelationType, EntityType, tuple[re.Pattern[str], ...], str], ...] = (
    (
        RelationType.PREFERS,
        EntityType.TOPIC,
        (
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+(?:really\s+)?(?:love(?:s)?|likes?|prefer(?:s)?|enjoy(?:s)?)\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+is\s+into\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
        ),
        "deterministic_preference_relation",
    ),
    (
        RelationType.AVOIDS,
        EntityType.TOPIC,
        (
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+(?:really\s+)?(?:hate|dislikes?|avoid(?:s)?)\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+do(?:es)?\s+not\s+like\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
        ),
        "deterministic_avoid_relation",
    ),
    (
        RelationType.WORKS_ON,
        EntityType.PROJECT,
        (
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+(?:am\s+)?working\s+on\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+works?\s+on\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+build(?:s|ing)?\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
        ),
        "deterministic_work_relation",
    ),
    (
        RelationType.DECIDED,
        EntityType.GOAL,
        (
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+decided\s+to\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+decided\s+on\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
        ),
        "deterministic_decision_relation",
    ),
    (
        RelationType.BLOCKED_BY,
        EntityType.TOPIC,
        (
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+(?:am\s+)?blocked\s+by\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
            re.compile(rf"^\s*(?:{_FIRST_PERSON_PREFIX})\s+(?:am\s+)?stuck\s+on\s+(?P<value>.+?)\s*\.?\s*$", re.IGNORECASE),
        ),
        "deterministic_blocker_relation",
    ),
)


class DeterministicGraphExtractionService:
    model_name = "deterministic-graph-v1"

    def __init__(self, *, primary_user_display_name: str | None = None) -> None:
        self.primary_user_display_name = _clean_phrase(primary_user_display_name or "") or None

    async def extract(self, record: MemoryRecord) -> GraphExtractionResult | None:
        if record.kind not in _SUPPORTED_KINDS:
            return None
        content = " ".join(str(record.content or "").split())
        if not content:
            return None

        name_result = self._extract_primary_user_identity(content)
        if name_result is not None:
            return name_result

        relation_result = self._extract_primary_user_relation(record, content)
        if relation_result is not None:
            return relation_result

        return GraphExtractionResult(
            should_extract=False,
            rationale="no_deterministic_graph_pattern",
        )

    async def close(self) -> None:
        return None

    def _extract_primary_user_identity(self, content: str) -> GraphExtractionResult | None:
        for pattern in _NAME_PATTERNS:
            match = pattern.match(content)
            if match is None:
                continue
            extracted_name = _clean_phrase(match.group("value"))
            if not extracted_name:
                continue
            return GraphExtractionResult(
                should_extract=True,
                rationale="deterministic_primary_user_identity",
                entities=[self._primary_user_entity(extracted_name=extracted_name)],
                relations=[],
            )
        return None

    def _extract_primary_user_relation(
        self,
        record: MemoryRecord,
        content: str,
    ) -> GraphExtractionResult | None:
        for relation_type, entity_type, patterns, rationale in _RELATION_PATTERNS:
            for pattern in patterns:
                match = pattern.match(content)
                if match is None:
                    continue
                value = _clean_object_phrase(match.group("value"))
                if not value:
                    continue
                return GraphExtractionResult(
                    should_extract=True,
                    rationale=rationale,
                    entities=[
                        self._primary_user_entity(),
                        ExtractedGraphEntity(
                            local_ref="fact_target",
                            entity_type=entity_type,
                            canonical_name=_canonicalize_object_name(value),
                            alias_values=[value] if value != _canonicalize_object_name(value) else [],
                        ),
                    ],
                    relations=[
                        ExtractedGraphRelation(
                            source_ref="primary_user",
                            target_ref="fact_target",
                            relation_type=relation_type,
                            fact=_ensure_terminal_period(content),
                            confidence=0.88,
                            valid_at=record.provenance.created_at,
                        )
                    ],
                )
        return None

    def _primary_user_entity(
        self,
        *,
        extracted_name: str | None = None,
    ) -> ExtractedGraphEntity:
        canonical_name = (
            self.primary_user_display_name
            or _clean_phrase(extracted_name or "")
            or _PRIMARY_USER_FALLBACK_NAME
        )
        alias_values: list[str] = []
        if canonical_name != _PRIMARY_USER_FALLBACK_NAME:
            alias_values.append(_PRIMARY_USER_FALLBACK_NAME)
        if extracted_name and extracted_name.casefold() != canonical_name.casefold():
            alias_values.append(extracted_name)
        return ExtractedGraphEntity(
            local_ref="primary_user",
            entity_type=EntityType.PERSON,
            canonical_name=canonical_name,
            identity_candidates=[
                GraphIdentityCandidate(
                    key_type=IdentityKeyType.EXTERNAL_ACCOUNT,
                    provider="cosmic",
                    raw_value="primary_user",
                    verified=True,
                )
            ],
            alias_values=list(dict.fromkeys(alias_values)),
            attributes={"scope": "primary_user"},
        )


def _clean_phrase(value: str) -> str:
    cleaned = " ".join(value.split()).strip(" \t\r\n\"'")
    return cleaned.rstrip(".,;:!?")


def _clean_object_phrase(value: str) -> str:
    cleaned = _clean_phrase(value)
    if not cleaned:
        return ""
    lowered = cleaned.casefold()
    for prefix in ("that ", "the fact that ", "to "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.casefold()
    return cleaned


def _canonicalize_object_name(value: str) -> str:
    if any(char.isupper() for char in value):
        return value
    words = [word for word in re.split(r"\s+", value) if word]
    if not words:
        return value
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _ensure_terminal_period(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return trimmed
    if trimmed.endswith((".", "!", "?")):
        return trimmed
    return f"{trimmed}."
