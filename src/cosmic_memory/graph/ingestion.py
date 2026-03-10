"""Helpers for graph ingestion from canonical memory records."""

from __future__ import annotations

from cosmic_memory.domain.models import MemoryRecord
from cosmic_memory.graph.models import GraphDocument


def graph_document_from_memory_record(record: MemoryRecord) -> GraphDocument | None:
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
    return document
