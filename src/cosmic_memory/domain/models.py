"""Pydantic models for the cosmic-memory service boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from cosmic_memory.domain.enums import MemoryKind, RecordStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryProvenance(BaseModel):
    source_kind: str = Field(description="Origin of this memory, such as gateway or agent.")
    source_id: str | None = Field(default=None, description="Origin-specific identifier.")
    created_by: str | None = Field(default=None, description="Actor that created this memory.")
    session_id: str | None = Field(default=None, description="Related session identifier.")
    task_id: str | None = Field(default=None, description="Related task identifier.")
    channel: str | None = Field(default=None, description="Origin channel, if any.")
    created_at: datetime = Field(default_factory=utc_now)


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid4().hex}")
    kind: MemoryKind
    title: str | None = None
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: MemoryProvenance
    status: RecordStatus = RecordStatus.ACTIVE
    version: int = 1
    supersedes: str | None = None
    superseded_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CanonicalMemorySnapshot(BaseModel):
    memory_id: str
    kind: MemoryKind
    status: RecordStatus
    version: int
    path: str
    content_hash: str
    token_count: int
    record: MemoryRecord


class WriteMemoryRequest(BaseModel):
    kind: MemoryKind
    content: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: MemoryProvenance


class SupersedeMemoryRequest(BaseModel):
    replacement: WriteMemoryRequest


class WriteCoreFactRequest(BaseModel):
    fact: str
    title: str | None = None
    canonical_key: str | None = None
    priority: int = 100
    always_include: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: MemoryProvenance

    def to_write_request(self) -> WriteMemoryRequest:
        metadata = dict(self.metadata)
        if self.canonical_key is not None:
            metadata["canonical_key"] = self.canonical_key
        metadata["priority"] = self.priority
        metadata["always_include"] = self.always_include
        return WriteMemoryRequest(
            kind=MemoryKind.CORE_FACT,
            content=self.fact,
            title=self.title,
            tags=self.tags,
            metadata=metadata,
            provenance=self.provenance,
        )


class CoreFactItem(BaseModel):
    memory_id: str
    title: str | None = None
    content: str
    canonical_key: str | None = None
    priority: int = 100
    always_include: bool = True
    tags: list[str] = Field(default_factory=list)


class CoreFactBlock(BaseModel):
    items: list[CoreFactItem] = Field(default_factory=list)
    rendered: str = ""


class EmbeddingCost(BaseModel):
    currency: str | None = None
    input_cost: float | None = None
    total_cost: float | None = None


class EmbeddingUsage(BaseModel):
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    cost: EmbeddingCost | None = None


class EmbeddingItem(BaseModel):
    index: int
    vector: list[float]
    dimensions: int


class GenerateEmbeddingsRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, min_length=1, max_length=4096)
    dimensions: int | None = Field(default=None, ge=128, le=2560)
    batch_size: int = Field(default=128, ge=1, le=512)
    max_parallel_requests: int = Field(default=4, ge=1, le=16)
    encoding_format: Literal["base64_int8", "base64_binary"] = "base64_int8"
    normalize: bool = True


class GenerateEmbeddingsResponse(BaseModel):
    model: str
    dimensions: int
    items: list[EmbeddingItem] = Field(default_factory=list)
    usage: EmbeddingUsage | None = None


class RecallItem(BaseModel):
    memory_id: str
    kind: MemoryKind
    title: str | None = None
    content: str
    score: float
    tags: list[str] = Field(default_factory=list)
    token_count: int | None = None


class PassiveRecallRequest(BaseModel):
    query: str
    kinds: list[MemoryKind] | None = None
    max_results: int = 8
    token_budget: int = 12_000


class PassiveRecallResponse(BaseModel):
    items: list[RecallItem] = Field(default_factory=list)
    total_token_count: int = 0


class IndexPointState(BaseModel):
    memory_id: str
    point_id: str | int | None = None
    status: RecordStatus | None = None
    content_hash: str | None = None


class IndexStatusResponse(BaseModel):
    enabled: bool
    collection_name: str | None = None
    canonical_count: int = 0
    registry_count: int = 0
    indexed_count: int = 0
    active_count: int = 0
    superseded_count: int = 0
    deleted_count: int = 0
    missing_from_registry: list[str] = Field(default_factory=list)
    stale_registry: list[str] = Field(default_factory=list)
    orphaned_registry: list[str] = Field(default_factory=list)
    missing_from_index: list[str] = Field(default_factory=list)
    stale_in_index: list[str] = Field(default_factory=list)
    orphaned_in_index: list[str] = Field(default_factory=list)


class IndexSyncResponse(BaseModel):
    enabled: bool
    collection_name: str | None = None
    mode: Literal["sync", "rebuild"]
    canonical_count: int = 0
    registry_upserts: int = 0
    registry_deletes: int = 0
    indexed_upserts: int = 0
    indexed_deletes: int = 0
    status: IndexStatusResponse


class GraphEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    memory_ids: list[str] = Field(default_factory=list)


class GraphRelation(BaseModel):
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    fact: str
    memory_ids: list[str] = Field(default_factory=list)
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


class ActiveRecallRequest(BaseModel):
    query: str
    kinds: list[MemoryKind] | None = None
    seed_memory_ids: list[str] = Field(default_factory=list)
    seed_entities: list[str] = Field(default_factory=list)
    max_results: int = 12
    max_hops: int = 2


class ActiveRecallResponse(BaseModel):
    items: list[RecallItem] = Field(default_factory=list)
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    search_plan: list[str] = Field(default_factory=list)


class HealthStatus(BaseModel):
    ok: bool
    service: str = "cosmic-memory"
    mode: str
