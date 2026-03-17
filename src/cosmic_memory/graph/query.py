"""Query framing for graph-assisted retrieval."""

from __future__ import annotations

import re

from cosmic_memory.graph.models import GraphIdentityCandidate, GraphQueryFrame
from cosmic_memory.graph.ontology import (
    INTENT_RELATION_MAP,
    IdentityKeyType,
    QueryIntent,
    RelationType,
)

EMAIL_REGEX = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")

RELATION_KEYWORDS: dict[RelationType, tuple[str, ...]] = {
    RelationType.HAS_IDENTITY_KEY: ("email", "contact", "phone", "username"),
    RelationType.WORKS_ON: ("work", "works", "working", "owner", "responsible", "project"),
    RelationType.PART_OF: ("part", "belongs", "component", "under"),
    RelationType.ATTENDED: (
        "attend",
        "attended",
        "studied",
        "school",
        "college",
        "university",
        "alma mater",
    ),
    RelationType.GRADUATED_FROM: (
        "graduate",
        "graduated",
        "graduation",
        "alumnus",
        "alumna",
        "degree",
    ),
    RelationType.MENTIONS: ("mention", "mentioned", "about", "reference"),
    RelationType.PREFERS: ("prefer", "preference", "default", "likes"),
    RelationType.AVOIDS: ("avoid", "avoids", "dislike", "skip"),
    RelationType.DECIDED: ("decide", "decided", "decision", "chosen", "choose"),
    RelationType.BLOCKED_BY: ("block", "blocked", "blocking", "blocker", "dependency", "depends"),
    RelationType.REMIND_AT: ("remind", "reminder", "due", "schedule"),
    RelationType.KNOWS: ("related", "relationship", "know", "with", "between"),
    RelationType.SUPERSEDES: ("replace", "replaced", "supersede", "updated"),
    RelationType.VALID_DURING: ("before", "after", "when", "last", "during", "current", "now"),
}


def build_query_frame(query: str) -> GraphQueryFrame:
    lower_query = query.casefold()
    query_tokens = _tokenize(query)
    intents: list[QueryIntent] = []
    if any(token in lower_query for token in {"who", "person", "email", "contact"}):
        intents.append(QueryIntent.ENTITY_LOOKUP)
    if any(token in lower_query for token in {"related", "relationship", "know", "with", "between"}):
        intents.append(QueryIntent.RELATION_LOOKUP)
    if any(token in lower_query for token in {"before", "after", "when", "last", "current", "now"}):
        intents.append(QueryIntent.TEMPORAL_LOOKUP)
    if any(
        token in lower_query
        for token in {"task", "project", "work", "decision", "decide", "decided", "choose", "chosen"}
    ):
        intents.append(QueryIntent.TASK_LOOKUP)
    if not intents:
        intents.append(QueryIntent.GENERIC)

    identity_candidates = [
        GraphIdentityCandidate(key_type=IdentityKeyType.EMAIL, raw_value=value)
        for value in EMAIL_REGEX.findall(query)
    ]

    entity_terms = sorted(query_tokens)
    allowed_relations = sorted(
        {
            relation
            for intent in intents
            for relation in INTENT_RELATION_MAP[intent]
        }
        | _relations_from_keywords(lower_query, query_tokens),
        key=lambda relation: relation.value,
    )
    prefer_current_state = any(token in lower_query for token in {"current", "currently", "now", "active"})

    return GraphQueryFrame(
        query=query,
        intents=intents,
        entity_terms=entity_terms,
        identity_candidates=identity_candidates,
        allowed_relations=allowed_relations,
        prefer_current_state=prefer_current_state,
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.casefold()) if token}


def _relations_from_keywords(lower_query: str, query_tokens: set[str]) -> set[RelationType]:
    matched: set[RelationType] = set()
    for relation, keywords in RELATION_KEYWORDS.items():
        if any(keyword in lower_query or keyword in query_tokens for keyword in keywords):
            matched.add(relation)
    return matched
