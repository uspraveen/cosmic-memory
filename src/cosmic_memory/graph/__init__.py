"""Graph primitives for identity resolution and traversal."""

from cosmic_memory.graph.base import GraphStore
from cosmic_memory.graph.dev_store import InMemoryGraphStore
from cosmic_memory.graph.identity import (
    build_identity_key,
    deterministic_identity_key_id,
    normalize_email,
    normalize_identity_value,
    normalize_name_variant,
)
from cosmic_memory.graph.ingestion import graph_document_from_memory_record
from cosmic_memory.graph.models import (
    GraphDocument,
    GraphDocumentEntity,
    GraphDocumentRelation,
    GraphEntityNode,
    GraphIdentityCandidate,
    GraphIdentityKey,
    GraphIngestResult,
    GraphQueryFrame,
    GraphRelationEdge,
    GraphSearchResult,
    IdentityResolutionResult,
)
from cosmic_memory.graph.neo4j_store import Neo4jGraphStore
from cosmic_memory.graph.ontology import EntityType, IdentityKeyType, QueryIntent, RelationType
from cosmic_memory.graph.query import build_query_frame

__all__ = [
    "EntityType",
    "GraphDocument",
    "GraphDocumentEntity",
    "GraphDocumentRelation",
    "GraphEntityNode",
    "GraphIdentityCandidate",
    "GraphIdentityKey",
    "GraphIngestResult",
    "GraphQueryFrame",
    "GraphRelationEdge",
    "GraphSearchResult",
    "GraphStore",
    "IdentityKeyType",
    "IdentityResolutionResult",
    "InMemoryGraphStore",
    "QueryIntent",
    "RelationType",
    "Neo4jGraphStore",
    "build_identity_key",
    "build_query_frame",
    "deterministic_identity_key_id",
    "graph_document_from_memory_record",
    "normalize_email",
    "normalize_identity_value",
    "normalize_name_variant",
]
