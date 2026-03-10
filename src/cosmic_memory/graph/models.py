"""Graph models for ontology, identity resolution, and traversal."""

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
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None


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
    entity_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    resolution_events: list[IdentityResolutionResult] = Field(default_factory=list)


class GraphQueryFrame(BaseModel):
    query: str
    intents: list[QueryIntent] = Field(default_factory=list)
    entity_terms: list[str] = Field(default_factory=list)
    identity_candidates: list[GraphIdentityCandidate] = Field(default_factory=list)
    allowed_relations: list[RelationType] = Field(default_factory=list)
    prefer_current_state: bool = True


class GraphSearchResult(BaseModel):
    entities: list[GraphEntityNode] = Field(default_factory=list)
    relations: list[GraphRelationEdge] = Field(default_factory=list)
    supporting_memory_ids: list[str] = Field(default_factory=list)
    search_plan: list[str] = Field(default_factory=list)
