"""Helpers for graph ingestion from canonical memory records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryRecord, utc_now
from cosmic_memory.graph.models import GraphDocument, GraphEpisode

if TYPE_CHECKING:
    from cosmic_memory.extraction.base import GraphExtractionService


def graph_document_from_memory_record(record: MemoryRecord) -> GraphDocument | None:
    from cosmic_memory.extraction.normalize import normalize_graph_document

    payload = record.metadata.get("graph_document")
    if payload is None:
        entities = record.metadata.get("entities")
        relations = record.metadata.get("relations")
        if not entities and not relations:
            return None
        payload = {
            "memory_id": record.memory_id,
            "entities": entities or [],
            "relations": relations or [],
            "source_text": record.content,
        }

    document = GraphDocument.model_validate(payload)
    if document.memory_id != record.memory_id:
        document.memory_id = record.memory_id
    if not document.source_text:
        document.source_text = record.content
    document.episode = _coerce_episode_for_record(record, document)
    normalized_document, _report = normalize_graph_document(document)
    return normalized_document


async def ensure_graph_document_for_record(
    record: MemoryRecord,
    *,
    extractor: "GraphExtractionService | None" = None,
) -> GraphDocument | None:
    from cosmic_memory.extraction.normalize import normalize_extraction_result

    document = graph_document_from_memory_record(record)
    if document is not None:
        document.episode = _coerce_episode_for_record(record, document)
        record.metadata["graph_document"] = document.model_dump(mode="json")
        return document

    if extractor is None or not should_extract_graph_for_record(record):
        return None

    extraction_result = await extractor.extract(record)
    if extraction_result is None:
        return None
    document, report = normalize_extraction_result(extraction_result, record=record)
    record.metadata["graph_extraction"] = {
        "mode": "auto",
        "model": getattr(extractor, "model_name", None),
        "extracted_at": utc_now().isoformat(),
        "input_entity_count": report.input_entity_count,
        "output_entity_count": report.output_entity_count,
        "input_relation_count": report.input_relation_count,
        "output_relation_count": report.output_relation_count,
        "merged_entity_count": report.merged_entity_count,
        "dropped_entity_count": report.dropped_entity_count,
        "dropped_relation_count": report.dropped_relation_count,
        "rationale": extraction_result.rationale,
    }
    if document is not None:
        document.episode = _coerce_episode_for_record(record, document)
        record.metadata["graph_document"] = document.model_dump(mode="json")
    return document


def should_extract_graph_for_kind(kind: MemoryKind) -> bool:
    return kind in {
        MemoryKind.CORE_FACT,
        MemoryKind.AGENT_NOTE,
        MemoryKind.SESSION_SUMMARY,
        MemoryKind.TASK_SUMMARY,
        MemoryKind.USER_DATA,
    }


def should_extract_graph_for_record(record: MemoryRecord) -> bool:
    if should_extract_graph_for_kind(record.kind):
        return True
    raw_override = record.metadata.get("extract_graph")
    if isinstance(raw_override, bool):
        return raw_override
    if isinstance(raw_override, str):
        normalized = raw_override.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return False


def _coerce_episode_for_record(record: MemoryRecord, document: GraphDocument) -> GraphEpisode:
    if document.episode is not None:
        episode = document.episode
        episode.memory_id = record.memory_id
        if not episode.source_type:
            episode.source_type = record.kind.value
        if not episode.provenance_source_kind:
            episode.provenance_source_kind = record.provenance.source_kind
        if not episode.provenance_source_id:
            episode.provenance_source_id = record.provenance.source_id
        if not episode.session_id:
            episode.session_id = record.provenance.session_id
        if not episode.task_id:
            episode.task_id = record.provenance.task_id
        if not episode.channel:
            episode.channel = record.provenance.channel
        if not episode.source_excerpt:
            episode.source_excerpt = _excerpt(record.content)
        return episode

    extraction_meta = record.metadata.get("graph_extraction") or {}
    extracted_at_raw = extraction_meta.get("extracted_at")
    extracted_at = utc_now()
    if isinstance(extracted_at_raw, str):
        try:
            extracted_at = GraphEpisode(extracted_at=extracted_at_raw, memory_id=record.memory_id, source_type=record.kind.value).extracted_at
        except Exception:
            extracted_at = utc_now()
    extraction_confidence = _episode_confidence(document)
    return GraphEpisode(
        memory_id=record.memory_id,
        source_type=record.kind.value,
        provenance_source_kind=record.provenance.source_kind,
        provenance_source_id=record.provenance.source_id,
        session_id=record.provenance.session_id,
        task_id=record.provenance.task_id,
        channel=record.provenance.channel,
        created_at=record.provenance.created_at,
        extracted_at=extracted_at,
        extraction_confidence=extraction_confidence,
        rationale=extraction_meta.get("rationale"),
        source_excerpt=_excerpt(record.content),
    )


def _episode_confidence(document: GraphDocument) -> float:
    if not document.relations:
        return 1.0
    total = sum(max(0.0, min(relation.confidence, 1.0)) for relation in document.relations)
    return total / len(document.relations)


def _excerpt(value: str, *, limit: int = 280) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."
