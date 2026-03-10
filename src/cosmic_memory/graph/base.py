"""Graph storage contract."""

from __future__ import annotations

from typing import Protocol

from cosmic_memory.graph.models import (
    GraphDocument,
    GraphEntityNode,
    GraphIdentityCandidate,
    GraphIngestResult,
    GraphQueryFrame,
    GraphSearchResult,
    IdentityResolutionResult,
)


class GraphStore(Protocol):
    async def ingest_document(self, document: GraphDocument) -> GraphIngestResult: ...

    async def remove_memory(self, memory_id: str) -> None: ...

    async def resolve_identity(
        self, candidate: GraphIdentityCandidate
    ) -> IdentityResolutionResult: ...

    async def passive_search(
        self,
        query_frame: GraphQueryFrame,
        *,
        max_entities: int = 5,
        max_relations: int = 8,
    ) -> GraphSearchResult: ...

    async def traverse(
        self,
        query_frame: GraphQueryFrame,
        *,
        seed_entity_ids: list[str] | None = None,
        max_hops: int = 2,
        max_entities: int = 10,
        max_relations: int = 12,
    ) -> GraphSearchResult: ...

    async def get_entity(self, entity_id: str) -> GraphEntityNode | None: ...
