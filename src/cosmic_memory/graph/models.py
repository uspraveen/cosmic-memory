"""Graph models for ontology, identity resolution, traversal, and provenance."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from cosmic_memory.domain.models import utc_now
from cosmic_memory.graph.ontology import EntityType, IdentityKeyType, QueryIntent, RelationType


class GraphIdentityCandidate(BaseModel):
    key_type: IdentityKeyType
    raw_value: str
    provider: str | None = None
    verified: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphIdentityKey(BaseModel):
    key_id: str
    key_type: IdentityKeyType
    normalized_value: str
    raw_values: list[str] = Field(default_factory=list)
    provider: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    memory_ids: list[str] = Field(default_factory=list)


class GraphEntityNode(BaseModel):
    entity_id: str = Field(default_factory=lambda: f"ent_{uuid4().hex}")
    entity_type: EntityType
    canonical_name: str
    alias_values: list[str] = Field(default_factory=list)
    identity_key_ids: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    memory_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resolution_state: Literal["canonical", "provisional"] = "canonical"


class GraphRelationEdge(BaseModel):
    relation_id: str
    relation_type: RelationType
    source_entity_id: str
    target_entity_id: str
    fact: str
    memory_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None
    invalidated_by_episode_id: str | None = None


class GraphEpisode(BaseModel):
    episode_id: str = Field(default_factory=lambda: f"ep_{uuid4().hex}")
    memory_id: str
    source_type: str
    provenance_source_kind: str | None = None
    provenance_source_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    channel: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    extracted_at: datetime = Field(default_factory=utc_now)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str | None = None
    source_excerpt: str | None = None
    produced_relation_ids: list[str] = Field(default_factory=list)
    invalidated_relation_ids: list[str] = Field(default_factory=list)


class GraphDocumentEntity(BaseModel):
    local_ref: str
    entity_type: EntityType
    canonical_name: str
    identity_candidates: list[GraphIdentityCandidate] = Field(default_factory=list)
    alias_values: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphDocumentRelation(BaseModel):
    source_ref: str
    target_ref: str
    relation_type: RelationType
    fact: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None


class GraphDocument(BaseModel):
    memory_id: str
    entities: list[GraphDocumentEntity] = Field(default_factory=list)
    relations: list[GraphDocumentRelation] = Field(default_factory=list)
    source_text: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    episode: GraphEpisode | None = None


class IdentityResolutionCandidate(BaseModel):
    entity_id: str
    reason: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class IdentityResolutionResult(BaseModel):
    status: Literal["exact_match", "candidate_match", "created_new", "no_match"]
    entity_id: str | None = None
    key: GraphIdentityKey | None = None
    candidates: list[IdentityResolutionCandidate] = Field(default_factory=list)


class GraphIngestResult(BaseModel):
    memory_id: str
    episode_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    invalidated_relation_ids: list[str] = Field(default_factory=list)
    resolution_events: list[IdentityResolutionResult] = Field(default_factory=list)


class GraphQueryFrame(BaseModel):
    query: str
    intents: list[QueryIntent] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    identity_candidates: list[GraphIdentityCandidate] = Field(default_factory=list)
    allowed_relations: list[RelationType] = Field(default_factory=list)
    prefer_current_state: bool = Field(
        default=True,
        description=(
            "When true, traversal and graph-assisted retrieval prefer active/current facts "
            "and suppress invalidated relations where possible. When false, callers are "
            "asking for broader historical traversal and invalidated relations may appear."
        ),
    )


class GraphSearchResult(BaseModel):
    entities: list[GraphEntityNode] = Field(default_factory=list)
    relations: list[GraphRelationEdge] = Field(default_factory=list)
    episodes: list[GraphEpisode] = Field(default_factory=list)
    seed_entity_ids: list[str] = Field(
        default_factory=list,
        description="Entity ids used as traversal seeds for this graph result.",
    )
    relation_distances: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "1-indexed hop distance from the nearest seed entity to each returned relation. "
            "A direct relation touching a seed entity has distance 1."
        ),
    )
    supporting_memory_ids: list[str] = Field(default_factory=list)
    search_plan: list[str] = Field(default_factory=list)


class GraphFactQuery(BaseModel):
    anchor_entity_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Undirected entity anchors. A relation matches if any anchored entity appears on "
            "either side of the relation."
        ),
    )
    source_entity_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Directional source-entity filter. Only relations whose source entity is in this "
            "set match."
        ),
    )
    target_entity_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Directional target-entity filter. Only relations whose target entity is in this "
            "set match."
        ),
    )
    relation_types: list[RelationType] = Field(
        default_factory=list,
        description="Allowed relation types for the fact lookup.",
    )
    active_only: bool = Field(
        default=True,
        description="When true, exclude invalidated, expired, or otherwise inactive relations.",
    )
    valid_at_or_after: datetime | None = Field(
        default=None,
        description=(
            "Optional lower temporal bound. Relations that ended entirely before this time do "
            "not match."
        ),
    )
    valid_at_or_before: datetime | None = Field(
        default=None,
        description=(
            "Optional upper temporal bound. Relations that start entirely after this time do "
            "not match."
        ),
    )
    max_results: int = Field(default=12, ge=1, le=100)
