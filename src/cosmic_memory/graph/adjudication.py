"""Internal entity adjudication for ambiguous graph writes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from cosmic_memory.graph.models import GraphDocumentEntity
from cosmic_memory.graph.ontology import EntityType


class EntityCandidateContext(BaseModel):
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    alias_values: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    resolution_state: str = "canonical"
    memory_count: int = 0
    relation_summaries: list[str] = Field(default_factory=list)
    similarity_score: float | None = None
    match_reasons: list[str] = Field(default_factory=list)


class EntityAdjudicationRequest(BaseModel):
    memory_id: str
    pending_entity: GraphDocumentEntity
    candidate_entities: list[EntityCandidateContext] = Field(default_factory=list)
    source_text: str | None = None
    utc_time_anchor: datetime
    local_time_anchor: datetime
    provenance_created_at: datetime
    timezone_name: str = "UTC"


class EntityAdjudicationDecision(BaseModel):
    decision: Literal["exact_match", "candidate_match", "created_new"]
    chosen_entity_id: str | None = None
    candidate_entity_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None


class EntityAdjudicationService(Protocol):
    model_name: str

    async def adjudicate(
        self,
        request: EntityAdjudicationRequest,
    ) -> EntityAdjudicationDecision: ...

    async def close(self) -> None: ...
