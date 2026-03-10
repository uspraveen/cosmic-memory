"""Structured graph extraction schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cosmic_memory.graph.models import GraphIdentityCandidate
from cosmic_memory.graph.ontology import EntityType, RelationType


class ExtractedGraphEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_ref: str = Field(description="Document-local identifier used by relations.")
    entity_type: EntityType = Field(description="Best ontology type for this entity.")
    canonical_name: str = Field(description="Best canonical display name for the entity.")
    identity_candidates: list[GraphIdentityCandidate] = Field(
        default_factory=list,
        description="Strong or weak identity candidates grounded in the text.",
    )
    alias_values: list[str] = Field(
        default_factory=list,
        description="Alternate grounded names, handles, or labels for the entity.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Short grounded attributes that help with later retrieval or disambiguation.",
    )


class ExtractedGraphRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(description="Source entity local_ref.")
    target_ref: str = Field(description="Target entity local_ref.")
    relation_type: RelationType = Field(description="Best ontology relation for this fact.")
    fact: str = Field(description="Short grounded fact sentence supporting the relation.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in this extracted relation.",
    )
    valid_at: datetime | None = Field(
        default=None,
        description="Absolute ISO datetime when the fact becomes true, if grounded.",
    )
    invalid_at: datetime | None = Field(
        default=None,
        description="Absolute ISO datetime when the fact stops being true, if grounded.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Absolute ISO datetime when a transient fact naturally expires, if grounded.",
    )


class GraphExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_extract: bool = Field(
        default=True,
        description="Whether the text contains useful graph-structured memory.",
    )
    rationale: str | None = Field(
        default=None,
        description="Short extraction rationale or skip reason.",
    )
    entities: list[ExtractedGraphEntity] = Field(
        default_factory=list,
        description="Entities grounded in the source memory.",
    )
    relations: list[ExtractedGraphRelation] = Field(
        default_factory=list,
        description="Typed relations grounded in the source memory.",
    )
