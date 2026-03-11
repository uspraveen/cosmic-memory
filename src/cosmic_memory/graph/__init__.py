"""Graph primitives for identity resolution and traversal."""

from cosmic_memory.graph.base import GraphStore
from cosmic_memory.graph.dev_store import InMemoryGraphStore
from cosmic_memory.graph.entity_index import (
    EntitySimilarityHit,
    EntitySimilarityIndex,
    InMemoryEntitySimilarityIndex,
)
from cosmic_memory.graph.entity_qdrant import QdrantEntitySimilarityIndex
from cosmic_memory.graph.identity import (
    build_identity_key,
    deterministic_identity_key_id,
    normalize_email,
    normalize_identity_value,
    normalize_name_variant,
)
from cosmic_memory.graph.ingestion import (
    ensure_graph_document_for_record,
    graph_document_from_memory_record,
    should_extract_graph_for_kind,
)
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
from cosmic_memory.graph.resolution import STRONG_KEY_TYPES, entity_allows_name_auto_merge

__all__ = [
    "EntityType",
    "EntitySimilarityHit",
    "EntitySimilarityIndex",
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
    "InMemoryEntitySimilarityIndex",
    "QueryIntent",
    "QdrantEntitySimilarityIndex",
    "RelationType",
    "Neo4jGraphStore",
    "build_identity_key",
    "build_query_frame",
    "deterministic_identity_key_id",
    "ensure_graph_document_for_record",
    "entity_allows_name_auto_merge",
    "graph_document_from_memory_record",
    "normalize_email",
    "normalize_identity_value",
    "normalize_name_variant",
    "should_extract_graph_for_kind",
    "STRONG_KEY_TYPES",
]
