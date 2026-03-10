"""Normalize and deduplicate extracted graph documents before ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from cosmic_memory.domain.models import MemoryRecord
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.graph.identity import build_identity_key
from cosmic_memory.graph.identity import normalize_name_variant
from cosmic_memory.graph.models import (
    GraphDocument,
    GraphDocumentEntity,
    GraphDocumentRelation,
)
from cosmic_memory.graph.ontology import IdentityKeyType, RelationType
from cosmic_memory.graph.resolution import (
    STRONG_KEY_TYPES,
    entity_allows_name_auto_merge,
    normalized_entity_names,
    preferred_canonical_name,
)


@dataclass(slots=True)
class GraphDocumentNormalizationReport:
    input_entity_count: int
    output_entity_count: int
    input_relation_count: int
    output_relation_count: int
    merged_entity_count: int
    dropped_entity_count: int
    dropped_relation_count: int


@dataclass(slots=True)
class _WorkingEntity:
    entity: GraphDocumentEntity
    strong_key_ids: set[str]
    name_keys: set[str]


def normalize_extraction_result(
    result: GraphExtractionResult,
    *,
    record: MemoryRecord,
) -> tuple[GraphDocument | None, GraphDocumentNormalizationReport]:
    if not result.should_extract:
        return None, GraphDocumentNormalizationReport(
            input_entity_count=len(result.entities),
            output_entity_count=0,
            input_relation_count=len(result.relations),
            output_relation_count=0,
            merged_entity_count=0,
            dropped_entity_count=len(result.entities),
            dropped_relation_count=len(result.relations),
        )

    return normalize_graph_document(
        GraphDocument(
            memory_id=record.memory_id,
            entities=[
                GraphDocumentEntity.model_validate(entity.model_dump(mode="python"))
                for entity in result.entities
            ],
            relations=[
                GraphDocumentRelation.model_validate(relation.model_dump(mode="python"))
                for relation in result.relations
            ],
            source_text=record.content,
        )
    )


def normalize_graph_document(
    document: GraphDocument,
) -> tuple[GraphDocument | None, GraphDocumentNormalizationReport]:
    ref_map: dict[str, str] = {}
    merged_entities: list[_WorkingEntity] = []
    input_entity_count = len(document.entities)
    input_relation_count = len(document.relations)
    dropped_entities = 0
    merged_count = 0

    for index, entity in enumerate(document.entities):
        normalized_entity = _normalize_entity(entity, index=index)
        if normalized_entity is None:
            dropped_entities += 1
            continue

        strong_key_ids = _strong_key_ids(normalized_entity)
        name_keys = normalized_entity_names(
            normalized_entity.canonical_name,
            normalized_entity.alias_values,
        )
        merge_target = _find_merge_target(
            merged_entities,
            normalized_entity,
            strong_key_ids=strong_key_ids,
            name_keys=name_keys,
        )
        if merge_target is None:
            merged_entities.append(
                _WorkingEntity(
                    entity=normalized_entity,
                    strong_key_ids=set(strong_key_ids),
                    name_keys=set(name_keys),
                )
            )
            ref_map[entity.local_ref] = normalized_entity.local_ref
            continue

        _merge_entity_into(
            target=merge_target,
            incoming=normalized_entity,
            strong_key_ids=strong_key_ids,
            name_keys=name_keys,
        )
        ref_map[entity.local_ref] = merge_target.entity.local_ref
        merged_count += 1

    deduped_relations: dict[str, GraphDocumentRelation] = {}
    dropped_relations = 0
    for relation in document.relations:
        normalized_relation = _normalize_relation(relation, ref_map=ref_map)
        if normalized_relation is None:
            dropped_relations += 1
            continue
        key = _relation_dedup_key(normalized_relation)
        existing = deduped_relations.get(key)
        if existing is None:
            deduped_relations[key] = normalized_relation
            continue
        existing.confidence = max(existing.confidence, normalized_relation.confidence)
        existing.valid_at = existing.valid_at or normalized_relation.valid_at
        existing.invalid_at = normalized_relation.invalid_at or existing.invalid_at
        existing.expires_at = existing.expires_at or normalized_relation.expires_at

    output_entities = [item.entity for item in merged_entities]
    output_relations = list(deduped_relations.values())
    report = GraphDocumentNormalizationReport(
        input_entity_count=input_entity_count,
        output_entity_count=len(output_entities),
        input_relation_count=input_relation_count,
        output_relation_count=len(output_relations),
        merged_entity_count=merged_count,
        dropped_entity_count=dropped_entities,
        dropped_relation_count=dropped_relations + max(input_relation_count - len(deduped_relations) - dropped_relations, 0),
    )
    if not output_entities and not output_relations:
        return None, report

    return (
        GraphDocument(
            memory_id=document.memory_id,
            entities=output_entities,
            relations=output_relations,
            source_text=document.source_text,
            created_at=document.created_at,
        ),
        report,
    )


def _normalize_entity(entity: GraphDocumentEntity, *, index: int) -> GraphDocumentEntity | None:
    canonical_name = _normalize_whitespace(entity.canonical_name)
    if not canonical_name:
        return None

    local_ref = _normalize_local_ref(entity.local_ref, index=index)
    alias_values = _dedup_aliases([*entity.alias_values, canonical_name])
    identity_candidates = _dedup_identity_candidates(entity.identity_candidates)
    attributes = {
        key: value
        for key, value in entity.attributes.items()
        if value is not None and key not in {"local_ref", "entity_type", "canonical_name"}
    }
    return GraphDocumentEntity(
        local_ref=local_ref,
        entity_type=entity.entity_type,
        canonical_name=canonical_name,
        identity_candidates=identity_candidates,
        alias_values=alias_values,
        attributes=attributes,
    )


def _normalize_relation(
    relation: GraphDocumentRelation,
    *,
    ref_map: dict[str, str],
) -> GraphDocumentRelation | None:
    source_ref = ref_map.get(relation.source_ref, relation.source_ref)
    target_ref = ref_map.get(relation.target_ref, relation.target_ref)
    if not source_ref or not target_ref or source_ref == target_ref:
        return None

    fact = _normalize_whitespace(relation.fact)
    if not fact:
        return None

    return GraphDocumentRelation(
        source_ref=source_ref,
        target_ref=target_ref,
        relation_type=relation.relation_type,
        fact=fact,
        confidence=max(0.0, min(relation.confidence, 1.0)),
        valid_at=relation.valid_at,
        invalid_at=relation.invalid_at,
        expires_at=relation.expires_at,
    )


def _find_merge_target(
    merged_entities: list[_WorkingEntity],
    entity: GraphDocumentEntity,
    *,
    strong_key_ids: set[str],
    name_keys: set[str],
) -> _WorkingEntity | None:
    for candidate in merged_entities:
        if candidate.entity.entity_type != entity.entity_type:
            continue
        if strong_key_ids and candidate.strong_key_ids & strong_key_ids:
            return candidate
        if (
            not strong_key_ids
            and entity_allows_name_auto_merge(entity.entity_type)
            and candidate.name_keys & name_keys
        ):
            return candidate
    return None


def _merge_entity_into(
    *,
    target: _WorkingEntity,
    incoming: GraphDocumentEntity,
    strong_key_ids: set[str],
    name_keys: set[str],
) -> None:
    target.entity.canonical_name = preferred_canonical_name(
        target.entity.canonical_name,
        incoming.canonical_name,
    )
    target.entity.alias_values = _dedup_aliases(
        [*target.entity.alias_values, *incoming.alias_values, incoming.canonical_name]
    )
    target.entity.identity_candidates = _dedup_identity_candidates(
        [*target.entity.identity_candidates, *incoming.identity_candidates]
    )
    target.entity.attributes.update(
        {key: value for key, value in incoming.attributes.items() if key not in target.entity.attributes}
    )
    target.strong_key_ids.update(strong_key_ids)
    target.name_keys.update(name_keys)


def _strong_key_ids(entity: GraphDocumentEntity) -> set[str]:
    key_ids: set[str] = set()
    for candidate in entity.identity_candidates:
        if candidate.key_type not in STRONG_KEY_TYPES:
            continue
        try:
            key_ids.add(build_identity_key(candidate).key_id)
        except ValueError:
            continue
    return key_ids


def _dedup_identity_candidates(identity_candidates):
    deduped: dict[str, object] = {}
    for candidate in identity_candidates:
        try:
            key = build_identity_key(candidate).key_id
        except ValueError:
            continue
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue
        if candidate.confidence > existing.confidence:
            deduped[key] = candidate
    return list(deduped.values())


def _dedup_aliases(values: list[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for value in values:
        normalized = _safe_name_key(value)
        if not normalized:
            continue
        existing = deduped.get(normalized)
        if existing is None or len(value.strip()) > len(existing):
            deduped[normalized] = _normalize_whitespace(value)
    return sorted(deduped.values())


def _normalize_local_ref(value: str, *, index: int) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")
    if normalized:
        return normalized
    return f"entity_{index + 1}"


def _relation_dedup_key(relation: GraphDocumentRelation) -> str:
    payload = "||".join(
        [
            relation.source_ref,
            relation.target_ref,
            relation.relation_type.value,
            _safe_name_key(relation.fact),
            relation.valid_at.isoformat() if relation.valid_at else "",
            relation.invalid_at.isoformat() if relation.invalid_at else "",
            relation.expires_at.isoformat() if relation.expires_at else "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _safe_name_key(value: str) -> str:
    try:
        return normalize_name_variant(value)
    except ValueError:
        return ""
