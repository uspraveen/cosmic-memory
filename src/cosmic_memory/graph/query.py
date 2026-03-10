"""Query framing for graph-assisted retrieval."""

from __future__ import annotations

import re

from cosmic_memory.graph.models import GraphIdentityCandidate, GraphQueryFrame
from cosmic_memory.graph.ontology import INTENT_RELATION_MAP, IdentityKeyType, QueryIntent

EMAIL_REGEX = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b")


def build_query_frame(query: str) -> GraphQueryFrame:
    lower_query = query.casefold()
    intents: list[QueryIntent] = []
    if any(token in lower_query for token in {"who", "person", "email", "contact"}):
        intents.append(QueryIntent.ENTITY_LOOKUP)
    if any(token in lower_query for token in {"related", "relationship", "know", "with", "between"}):
        intents.append(QueryIntent.RELATION_LOOKUP)
    if any(token in lower_query for token in {"before", "after", "when", "last", "current", "now"}):
        intents.append(QueryIntent.TEMPORAL_LOOKUP)
    if any(token in lower_query for token in {"task", "project", "work", "decision"}):
        intents.append(QueryIntent.TASK_LOOKUP)
    if not intents:
        intents.append(QueryIntent.GENERIC)

    identity_candidates = [
        GraphIdentityCandidate(key_type=IdentityKeyType.EMAIL, raw_value=value)
        for value in EMAIL_REGEX.findall(query)
    ]

    entity_terms = sorted(_tokenize(query))
    allowed_relations = sorted(
        {relation for intent in intents for relation in INTENT_RELATION_MAP[intent]},
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
