"""Helpers for graph ingestion from canonical memory records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryRecord, utc_now
from cosmic_memory.graph.models import GraphDocument

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
        record.metadata["graph_document"] = document.model_dump(mode="json")
        return document

    if extractor is None or not should_extract_graph_for_kind(record.kind):
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
