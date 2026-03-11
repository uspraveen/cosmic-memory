"""Entity-level similarity index used for candidate generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from cosmic_memory.domain.models import GenerateEmbeddingsRequest
from cosmic_memory.embeddings.base import EmbeddingService
from cosmic_memory.graph.models import GraphEntityNode
from cosmic_memory.graph.ontology import EntityType


class EntitySimilarityHit(BaseModel):
    entity_id: str
    score: float
    entity_type: EntityType
    canonical_name: str
    alias_values: list[str] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    resolution_state: str = "canonical"


class EntitySimilarityIndex(Protocol):
    async def ensure_ready(self) -> None: ...

    async def sync_entity(self, entity: GraphEntityNode) -> None: ...

    async def sync_entities(self, entities: Sequence[GraphEntityNode]) -> None: ...

    async def delete_entities(self, entity_ids: Sequence[str]) -> None: ...

    async def search(
        self,
        query: str,
        *,
        entity_types: Sequence[EntityType] | None = None,
        limit: int = 8,
    ) -> list[EntitySimilarityHit]: ...

    async def close(self) -> None: ...


class InMemoryEntitySimilarityIndex:
    """Small in-memory entity similarity index for tests and local development."""

    def __init__(self, *, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._entities: dict[str, GraphEntityNode] = {}
        self._vectors: dict[str, list[float]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def sync_entity(self, entity: GraphEntityNode) -> None:
        await self.sync_entities([entity])

    async def sync_entities(self, entities: Sequence[GraphEntityNode]) -> None:
        if not entities:
            return
        texts = [entity_similarity_text(entity) for entity in entities]
        result = await self.embedding_service.generate(
            GenerateEmbeddingsRequest(
                texts=texts,
                dimensions=self.embedding_service.dimensions,
                batch_size=min(max(len(texts), 1), 64),
                max_parallel_requests=4,
                normalize=True,
            )
        )
        vectors = [item.vector for item in sorted(result.items, key=lambda item: item.index)]
        for entity, vector in zip(entities, vectors, strict=True):
            self._entities[entity.entity_id] = entity.model_copy(deep=True)
            self._vectors[entity.entity_id] = vector

    async def delete_entities(self, entity_ids: Sequence[str]) -> None:
        for entity_id in entity_ids:
            self._entities.pop(entity_id, None)
            self._vectors.pop(entity_id, None)

    async def search(
        self,
        query: str,
        *,
        entity_types: Sequence[EntityType] | None = None,
        limit: int = 8,
    ) -> list[EntitySimilarityHit]:
        if not query.strip() or not self._entities:
            return []
        result = await self.embedding_service.generate(
            GenerateEmbeddingsRequest(
                texts=[query],
                dimensions=self.embedding_service.dimensions,
                batch_size=1,
                max_parallel_requests=1,
                normalize=True,
            )
        )
        query_vector = result.items[0].vector
        allowed_types = set(entity_types or [])
        ranked: list[EntitySimilarityHit] = []
        for entity_id, entity in self._entities.items():
            if allowed_types and entity.entity_type not in allowed_types:
                continue
            vector_score = _cosine_similarity(query_vector, self._vectors.get(entity_id, []))
            lexical_score = _lexical_similarity(query, entity)
            score = (vector_score * 0.75) + (lexical_score * 0.25)
            if score <= 0:
                continue
            ranked.append(
                EntitySimilarityHit(
                    entity_id=entity.entity_id,
                    score=score,
                    entity_type=entity.entity_type,
                    canonical_name=entity.canonical_name,
                    alias_values=entity.alias_values,
                    memory_ids=entity.memory_ids,
                    resolution_state=entity.resolution_state,
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    async def close(self) -> None:
        return None


def entity_similarity_text(entity: GraphEntityNode) -> str:
    return entity_similarity_text_parts(
        entity_type=entity.entity_type,
        canonical_name=entity.canonical_name,
        alias_values=entity.alias_values,
        attributes=entity.attributes,
    )


def entity_similarity_text_parts(
    *,
    entity_type: EntityType,
    canonical_name: str,
    alias_values: Sequence[str],
    attributes: dict[str, object],
) -> str:
    attribute_values = _flatten_attribute_values(attributes)
    return " | ".join(
        value
        for value in [
            entity_type.value,
            canonical_name,
            " ".join(alias_values),
            " ".join(attribute_values),
        ]
        if value
    )


def _flatten_attribute_values(attributes: dict[str, object]) -> list[str]:
    values: list[str] = []
    for value in attributes.values():
        if value is None:
            continue
        if isinstance(value, str):
            values.append(value)
            continue
        if isinstance(value, (int, float, bool)):
            values.append(str(value))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            values.extend(str(item) for item in value if item is not None)
    return values


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_similarity(query: str, entity: GraphEntityNode) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    entity_tokens = _tokenize(" ".join([entity.canonical_name, *entity.alias_values]))
    overlap = len(query_tokens & entity_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(query_tokens)


def _tokenize(text: str) -> set[str]:
    return {token for token in text.casefold().replace("|", " ").split() if token}
