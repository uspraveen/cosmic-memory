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
        RelationType.BLOCKED_BY,
        RelationType.DECIDED,
        RelationType.MENTIONS,
    },
}
