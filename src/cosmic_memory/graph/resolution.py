"""Shared graph identity and entity merge rules."""

from __future__ import annotations

from cosmic_memory.graph.identity import normalize_name_variant
from cosmic_memory.graph.ontology import EntityType, IdentityKeyType

STRONG_KEY_TYPES = {
    IdentityKeyType.EMAIL,
    IdentityKeyType.PHONE,
    IdentityKeyType.EXTERNAL_ACCOUNT,
}

# Non-person entities usually need cross-memory continuity from stable names.
AUTO_MERGE_NAME_ENTITY_TYPES = {
    EntityType.ORGANIZATION,
    EntityType.PROJECT,
    EntityType.TASK,
    EntityType.GOAL,
    EntityType.REMINDER,
    EntityType.TOPIC,
    EntityType.SESSION,
}

SIMILARITY_COMPATIBLE_ENTITY_TYPES: dict[EntityType, set[EntityType]] = {
    EntityType.PERSON: {EntityType.PERSON},
    EntityType.ORGANIZATION: {EntityType.ORGANIZATION},
    EntityType.PROJECT: {EntityType.PROJECT, EntityType.TOPIC},
    EntityType.TASK: {EntityType.TASK, EntityType.REMINDER, EntityType.GOAL, EntityType.TOPIC},
    EntityType.REMINDER: {EntityType.REMINDER, EntityType.TASK, EntityType.GOAL, EntityType.TOPIC},
    EntityType.GOAL: {EntityType.GOAL, EntityType.TASK, EntityType.REMINDER, EntityType.TOPIC},
    EntityType.TOPIC: {
        EntityType.TOPIC,
        EntityType.TASK,
        EntityType.REMINDER,
        EntityType.GOAL,
        EntityType.PROJECT,
        EntityType.ARTIFACT,
    },
    EntityType.ARTIFACT: {EntityType.ARTIFACT, EntityType.TOPIC, EntityType.PROJECT},
    EntityType.EVENT: {EntityType.EVENT, EntityType.SESSION},
    EntityType.SESSION: {EntityType.SESSION, EntityType.EVENT},
    EntityType.IDENTITY_KEY: {EntityType.IDENTITY_KEY},
}


def entity_allows_name_auto_merge(entity_type: EntityType) -> bool:
    return entity_type in AUTO_MERGE_NAME_ENTITY_TYPES


def similarity_candidate_types(entity_type: EntityType) -> set[EntityType]:
    return set(SIMILARITY_COMPATIBLE_ENTITY_TYPES.get(entity_type, {entity_type}))


def normalized_entity_names(
    canonical_name: str,
    alias_values: list[str] | tuple[str, ...],
) -> set[str]:
    values = [canonical_name, *alias_values]
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(normalize_name_variant(value))
        except ValueError:
            continue
    return normalized


def preferred_canonical_name(current_name: str, candidate_name: str) -> str:
    current = _safe_name_key(current_name)
    candidate = _safe_name_key(candidate_name)
    if not current:
        return candidate_name
    if not candidate:
        return current_name
    current_score = (_token_count(current), len(current))
    candidate_score = (_token_count(candidate), len(candidate))
    if candidate_score > current_score:
        return candidate_name
    return current_name


def _safe_name_key(value: str) -> str:
    try:
        return normalize_name_variant(value)
    except ValueError:
        return ""


def _token_count(value: str) -> int:
    return len([token for token in value.split(" ") if token])
