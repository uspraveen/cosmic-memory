"""Graph ontology primitives for Cosmic memory."""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TASK = "task"
    ARTIFACT = "artifact"
    PREFERENCE = "preference"
    GOAL = "goal"
    REMINDER = "reminder"
    TOPIC = "topic"
    EVENT = "event"
    SESSION = "session"
    IDENTITY_KEY = "identity_key"


class IdentityKeyType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    EXTERNAL_ACCOUNT = "external_account"
    USERNAME = "username"
    NAME_VARIANT = "name_variant"


class RelationType(StrEnum):
    HAS_IDENTITY_KEY = "has_identity_key"
    WORKS_ON = "works_on"
    PART_OF = "part_of"
    ATTENDED = "attended"
    GRADUATED_FROM = "graduated_from"
    MENTIONS = "mentions"
    PREFERS = "prefers"
    AVOIDS = "avoids"
    DECIDED = "decided"
    BLOCKED_BY = "blocked_by"
    REMIND_AT = "remind_at"
    KNOWS = "knows"
    SUPERSEDES = "supersedes"
    VALID_DURING = "valid_during"


class QueryIntent(StrEnum):
    GENERIC = "generic"
    ENTITY_LOOKUP = "entity_lookup"
    RELATION_LOOKUP = "relation_lookup"
    TEMPORAL_LOOKUP = "temporal_lookup"
    TASK_LOOKUP = "task_lookup"


INTENT_RELATION_MAP: dict[QueryIntent, set[RelationType]] = {
    QueryIntent.GENERIC: {
        RelationType.WORKS_ON,
        RelationType.PART_OF,
        RelationType.ATTENDED,
        RelationType.GRADUATED_FROM,
        RelationType.MENTIONS,
        RelationType.DECIDED,
        RelationType.KNOWS,
    },
    QueryIntent.ENTITY_LOOKUP: {
        RelationType.HAS_IDENTITY_KEY,
        RelationType.KNOWS,
        RelationType.MENTIONS,
    },
    QueryIntent.RELATION_LOOKUP: {
        RelationType.KNOWS,
        RelationType.WORKS_ON,
        RelationType.PART_OF,
        RelationType.ATTENDED,
        RelationType.GRADUATED_FROM,
        RelationType.DECIDED,
        RelationType.BLOCKED_BY,
    },
    QueryIntent.TEMPORAL_LOOKUP: {
        RelationType.REMIND_AT,
        RelationType.VALID_DURING,
        RelationType.SUPERSEDES,
        RelationType.DECIDED,
    },
    QueryIntent.TASK_LOOKUP: {
        RelationType.WORKS_ON,
        RelationType.PART_OF,
        RelationType.ATTENDED,
        RelationType.GRADUATED_FROM,
        RelationType.BLOCKED_BY,
        RelationType.DECIDED,
        RelationType.MENTIONS,
    },
}


_RELATION_COMPATIBILITY_MAP: dict[RelationType, set[RelationType]] = {
    RelationType.HAS_IDENTITY_KEY: {RelationType.HAS_IDENTITY_KEY},
    RelationType.WORKS_ON: {RelationType.WORKS_ON, RelationType.PART_OF},
    RelationType.PART_OF: {RelationType.PART_OF, RelationType.WORKS_ON},
    RelationType.ATTENDED: {
        RelationType.ATTENDED,
        RelationType.GRADUATED_FROM,
        RelationType.PART_OF,
    },
    RelationType.GRADUATED_FROM: {
        RelationType.GRADUATED_FROM,
        RelationType.ATTENDED,
        RelationType.PART_OF,
    },
    RelationType.MENTIONS: {RelationType.MENTIONS},
    RelationType.PREFERS: {RelationType.PREFERS, RelationType.AVOIDS},
    RelationType.AVOIDS: {RelationType.AVOIDS, RelationType.PREFERS},
    RelationType.DECIDED: {RelationType.DECIDED, RelationType.SUPERSEDES},
    RelationType.BLOCKED_BY: {RelationType.BLOCKED_BY},
    RelationType.REMIND_AT: {RelationType.REMIND_AT, RelationType.VALID_DURING},
    RelationType.KNOWS: {RelationType.KNOWS},
    RelationType.SUPERSEDES: {RelationType.SUPERSEDES, RelationType.DECIDED},
    RelationType.VALID_DURING: {RelationType.VALID_DURING, RelationType.REMIND_AT},
}


def compatible_relation_types(relation_type: RelationType) -> set[RelationType]:
    return set(_RELATION_COMPATIBILITY_MAP.get(relation_type, {relation_type}))
