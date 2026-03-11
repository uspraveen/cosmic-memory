"""Agent-facing control surface for orchestrator memory use."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallResponse,
    CoreFactBlock,
    GraphEntity,
    GraphRelation,
    PassiveRecallResponse,
    RecallItem,
)
from cosmic_memory.graph import EntityType, IdentityKeyType, QueryIntent, RelationType
from cosmic_memory.retrieval import build_query_signals

CURRENT_STATE_RELATION_TYPES = {
    RelationType.WORKS_ON.value,
    RelationType.PART_OF.value,
    RelationType.PREFERS.value,
    RelationType.AVOIDS.value,
    RelationType.BLOCKED_BY.value,
    RelationType.REMIND_AT.value,
}
TEMPORAL_RELATION_TYPES = {
    RelationType.REMIND_AT.value,
    RelationType.SUPERSEDES.value,
    RelationType.VALID_DURING.value,
    RelationType.DECIDED.value,
}


class MemoryToolDescription(BaseModel):
    name: str
    when_to_use: str
    output_shape: str


class SchemaContextResponse(BaseModel):
    graph_available: bool
    memory_kinds: list[str] = Field(default_factory=list)
    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    query_intents: list[str] = Field(default_factory=list)
    tools: list[MemoryToolDescription] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class QueryFrameSummary(BaseModel):
    intents: list[str] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    identity_candidates: list[str] = Field(default_factory=list)
    allowed_relations: list[str] = Field(default_factory=list)
    prefer_current_state: bool = False


class MemoryQueryPlanRequest(BaseModel):
    query: str
    max_hops: int = Field(default=2, ge=1, le=4)


class MemoryQueryPlanResponse(BaseModel):
    query: str
    recommended_mode: Literal["passive", "active", "hybrid"]
    include_schema_context: bool = False
    suggested_max_hops: int = 1
    tool_sequence: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    frame: QueryFrameSummary


class ResolveIdentityRequest(BaseModel):
    value: str
    key_type: Literal["email", "phone", "external_account", "username", "name_variant"] = "email"
    provider: str | None = None
    verified: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ResolvedEntityRef(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    memory_ids: list[str] = Field(default_factory=list)


class ResolveIdentityCandidate(BaseModel):
    entity_id: str
    reason: str
    confidence: float
    name: str | None = None
    entity_type: str | None = None


class ResolveIdentityResponse(BaseModel):
    graph_available: bool
    status: Literal["exact_match", "candidate_match", "created_new", "no_match"]
    normalized_key_id: str | None = None
    normalized_value: str | None = None
    entity: ResolvedEntityRef | None = None
    candidates: list[ResolveIdentityCandidate] = Field(default_factory=list)


class GraphFactItem(BaseModel):
    relation_id: str
    relation_type: str
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    fact: str
    memory_ids: list[str] = Field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


class CurrentStateRequest(BaseModel):
    query: str
    max_results: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=4)
    include_supporting_memories: bool = True
    include_diagnostics: bool = False


class CurrentStateResponse(BaseModel):
    query: str
    facts: list[GraphFactItem] = Field(default_factory=list)
    entities: list[GraphEntity] = Field(default_factory=list)
    supporting_memories: list[RecallItem] = Field(default_factory=list)
    search_plan: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] | None = None


class TemporalFactsRequest(BaseModel):
    query: str
    max_results: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=4)
    include_supporting_memories: bool = True
    include_diagnostics: bool = False


class TemporalFactsResponse(BaseModel):
    query: str
    facts: list[GraphFactItem] = Field(default_factory=list)
    entities: list[GraphEntity] = Field(default_factory=list)
    supporting_memories: list[RecallItem] = Field(default_factory=list)
    search_plan: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] | None = None


class MemoryBriefRequest(BaseModel):
    query: str
    token_budget: int = Field(default=4_000, ge=256, le=20_000)
    passive_max_results: int = Field(default=6, ge=1, le=20)
    active_max_results: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=4)
    include_core_facts: bool = False
    include_diagnostics: bool = False


class MemoryBriefResponse(BaseModel):
    plan: MemoryQueryPlanResponse
    core_facts: CoreFactBlock | None = None
    passive: PassiveRecallResponse
    active: ActiveRecallResponse | None = None
    current_state: CurrentStateResponse | None = None
    temporal: TemporalFactsResponse | None = None
    findings: list[str] = Field(default_factory=list)
    supporting_memory_ids: list[str] = Field(default_factory=list)


def build_schema_context(*, graph_available: bool) -> SchemaContextResponse:
    tools = [
        MemoryToolDescription(
            name="passive_search",
            when_to_use="Fast default recall on every query before heavier graph work.",
            output_shape="Ranked memories under a fixed token budget.",
        ),
        MemoryToolDescription(
            name="resolve_identity",
            when_to_use="Normalize emails, usernames, phones, or aliases before traversal.",
            output_shape="Deterministic identity key plus exact or candidate entity matches.",
        ),
        MemoryToolDescription(
            name="current_state",
            when_to_use="Ask for what is active right now: blockers, preferences, owners, reminders, or current work.",
            output_shape="Current graph facts plus supporting memories.",
        ),
        MemoryToolDescription(
            name="temporal_facts",
            when_to_use="Ask about before/after/when/history/timeline questions.",
            output_shape="Time-aware graph facts plus supporting memories.",
        ),
        MemoryToolDescription(
            name="active_recall",
            when_to_use="Use for relationship-heavy or multi-hop recall that passive retrieval cannot answer cleanly.",
            output_shape="Graph entities, relations, and supporting memories.",
        ),
        MemoryToolDescription(
            name="memory_brief",
            when_to_use="Use when the orchestrator wants a structured memory investigation bundle instead of composing calls manually.",
            output_shape="Plan, passive recall, active recall, current/temporal facts, and distilled findings.",
        ),
    ]
    notes = [
        "Canonical Markdown is the memory source of truth; Qdrant and Neo4j are derived retrieval layers.",
        "Passive retrieval is Qdrant-first and graph-assisted; do not put LLMs in the passive hot path.",
        "Use schema_context before complex memory planning so the orchestrator sees the ontology and tool contracts.",
        "Use typed tools instead of generating raw Cypher from the model.",
    ]
    if not graph_available:
        notes.append("Graph traversal is currently unavailable; active graph tools will degrade to memory-only behavior.")
    return SchemaContextResponse(
        graph_available=graph_available,
        memory_kinds=[kind.value for kind in MemoryKind],
        entity_types=[entity_type.value for entity_type in EntityType],
        relation_types=[relation_type.value for relation_type in RelationType],
        query_intents=[intent.value for intent in QueryIntent],
        tools=tools,
        notes=notes,
    )


def build_memory_query_plan(
    request: MemoryQueryPlanRequest,
    *,
    graph_available: bool,
) -> MemoryQueryPlanResponse:
    signals = build_query_signals(request.query)
    deep_reasoning_hint = any(
        token in signals.query_lower
        for token in {"why", "how", "compare", "difference", "conflict", "contradict", "relationship", "between"}
    )
    graph_heavy = (
        graph_available
        and (
            signals.relationship_hint
            or signals.temporal_hint
            or signals.current_state_hint
            or signals.identity_hint
            or deep_reasoning_hint
        )
    )
    recommended_mode: Literal["passive", "active", "hybrid"]
    if graph_heavy and (signals.temporal_hint or deep_reasoning_hint):
        recommended_mode = "active"
    elif graph_heavy:
        recommended_mode = "hybrid"
    else:
        recommended_mode = "passive"

    tool_sequence: list[str] = []
    rationale: list[str] = []
    include_schema_context = graph_heavy
    if include_schema_context:
        tool_sequence.append("schema_context")
        rationale.append("The query needs ontology-aware planning before graph operations.")

    tool_sequence.append("passive_search")
    rationale.append("Start with fast passive recall to anchor the search in canonical memories.")

    if graph_available and signals.identity_hint:
        tool_sequence.append("resolve_identity")
        rationale.append("The query includes an identity or exact key that should be normalized before traversal.")
    if graph_available and signals.current_state_hint:
        tool_sequence.append("current_state")
        rationale.append("The query asks for active or current memory state.")
    elif graph_available and signals.temporal_hint:
        tool_sequence.append("temporal_facts")
        rationale.append("The query has explicit temporal intent and should prefer time-aware graph facts.")
    elif graph_available and (signals.relationship_hint or signals.task_hint or deep_reasoning_hint):
        tool_sequence.append("active_recall")
        rationale.append("The query is relation-heavy and likely needs graph traversal.")

    if graph_available and recommended_mode in {"active", "hybrid"} and "memory_brief" not in tool_sequence:
        tool_sequence.append("memory_brief")
        rationale.append("Bundle passive and graph evidence into a structured memory brief for the orchestrator.")

    suggested_max_hops = 1
    if recommended_mode == "hybrid":
        suggested_max_hops = min(2, request.max_hops)
    elif recommended_mode == "active":
        suggested_max_hops = min(max(2, request.max_hops), 3)

    frame = signals.query_frame
    return MemoryQueryPlanResponse(
        query=request.query,
        recommended_mode=recommended_mode,
        include_schema_context=include_schema_context,
        suggested_max_hops=suggested_max_hops,
        tool_sequence=tool_sequence,
        rationale=rationale,
        frame=QueryFrameSummary(
            intents=[intent.value for intent in frame.intents],
            entity_terms=frame.entity_terms,
            identity_candidates=[candidate.raw_value for candidate in frame.identity_candidates],
            allowed_relations=[relation.value for relation in frame.allowed_relations],
            prefer_current_state=frame.prefer_current_state,
        ),
    )


def graph_facts_from_active_response(response: ActiveRecallResponse) -> list[GraphFactItem]:
    entity_map = {entity.entity_id: entity for entity in response.entities}
    facts: list[GraphFactItem] = []
    for relation in response.relations:
        source = entity_map.get(relation.source_entity_id)
        target = entity_map.get(relation.target_entity_id)
        facts.append(
            GraphFactItem(
                relation_id=relation.relation_id,
                relation_type=relation.relation_type,
                source_entity_id=relation.source_entity_id,
                source_entity_name=source.name if source is not None else relation.source_entity_id,
                target_entity_id=relation.target_entity_id,
                target_entity_name=target.name if target is not None else relation.target_entity_id,
                fact=relation.fact,
                memory_ids=relation.memory_ids,
                valid_at=relation.valid_at,
                invalid_at=relation.invalid_at,
            )
        )
    return facts


def filter_current_state_facts(facts: list[GraphFactItem]) -> list[GraphFactItem]:
    now = datetime.now(timezone.utc)
    filtered: list[GraphFactItem] = []
    for fact in facts:
        if fact.invalid_at is not None and fact.invalid_at <= now:
            continue
        if fact.valid_at is not None and fact.valid_at > now:
            continue
        if fact.relation_type in CURRENT_STATE_RELATION_TYPES or fact.valid_at is not None:
            filtered.append(fact)
    return filtered


def filter_temporal_facts(facts: list[GraphFactItem]) -> list[GraphFactItem]:
    filtered: list[GraphFactItem] = []
    for fact in facts:
        if (
            fact.relation_type in TEMPORAL_RELATION_TYPES
            or fact.valid_at is not None
            or fact.invalid_at is not None
        ):
            filtered.append(fact)
    return filtered


def build_brief_findings(
    *,
    current_state: CurrentStateResponse | None,
    temporal: TemporalFactsResponse | None,
    active: ActiveRecallResponse | None,
    passive: PassiveRecallResponse,
) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()

    for fact in (current_state.facts if current_state else [])[:3]:
        finding = f"Current: {fact.source_entity_name} {fact.relation_type} {fact.target_entity_name}."
        if finding not in seen:
            findings.append(finding)
            seen.add(finding)

    for fact in (temporal.facts if temporal else [])[:3]:
        marker = fact.valid_at.isoformat() if fact.valid_at else "undated"
        finding = f"Timeline: {fact.source_entity_name} {fact.relation_type} {fact.target_entity_name} ({marker})."
        if finding not in seen:
            findings.append(finding)
            seen.add(finding)

    for fact in graph_facts_from_active_response(active)[:3] if active is not None else []:
        finding = f"Graph: {fact.fact}"
        if finding not in seen:
            findings.append(finding)
            seen.add(finding)

    for item in passive.items[:3]:
        finding = f"Memory: {item.title or item.memory_id}."
        if finding not in seen:
            findings.append(finding)
            seen.add(finding)

    return findings
