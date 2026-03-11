"""Internal fact adjudication for ambiguous graph fact writes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from cosmic_memory.graph.models import GraphEpisode, GraphRelationEdge
from cosmic_memory.graph.ontology import RelationType


class PendingFactContext(BaseModel):
    relation_type: RelationType
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    fact: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None


class FactCandidateContext(BaseModel):
    relation_id: str
    relation_type: RelationType
    source_entity_id: str
    source_entity_name: str
    target_entity_id: str
    target_entity_name: str
    fact: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    memory_count: int = Field(default=0, ge=0)
    episode_count: int = Field(default=0, ge=0)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    expires_at: datetime | None = None
    active: bool = True
    retrieval_reason: str | None = None


class FactAdjudicationRequest(BaseModel):
    memory_id: str
    episode: GraphEpisode
    pending_fact: PendingFactContext
    candidate_facts: list[FactCandidateContext] = Field(default_factory=list)
    source_text: str | None = None
    utc_time_anchor: datetime
    local_time_anchor: str
    provenance_created_at: datetime
    timezone_name: str


class FactAdjudicationDecision(BaseModel):
    decision: Literal["keep_both", "merge_with_existing", "invalidate_existing", "discard_new"]
    chosen_relation_id: str | None = None
    invalidated_relation_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str | None = None


class FactAdjudicationService(Protocol):
    async def adjudicate(
        self,
        request: FactAdjudicationRequest,
    ) -> FactAdjudicationDecision: ...

    async def close(self) -> None: ...


def exact_fact_signature(relation: GraphRelationEdge | PendingFactContext) -> tuple[str, str, str, str]:
    return (
        relation.source_entity_id,
        relation.target_entity_id,
        relation.relation_type.value,
        " ".join(relation.fact.casefold().split()),
    )
