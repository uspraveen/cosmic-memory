import asyncio

from cosmic_memory.domain.models import (
    EmbeddingItem,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
)
from cosmic_memory.graph import (
    EntityAdjudicationDecision,
    EntityType,
    GraphDocument,
    GraphDocumentEntity,
    GraphDocumentRelation,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryEntitySimilarityIndex,
    InMemoryGraphStore,
    RelationType,
    build_query_frame,
)


class SemanticTestEmbeddingService:
    model_name = "semantic-test"
    dimensions = 128

    async def generate(self, request: GenerateEmbeddingsRequest) -> GenerateEmbeddingsResponse:
        items: list[EmbeddingItem] = []
        for index, text in enumerate(request.texts):
            items.append(
                EmbeddingItem(
                    index=index,
                    vector=self._embed(text),
                    dimensions=self.dimensions,
                )
            )
        return GenerateEmbeddingsResponse(
            model=self.model_name,
            dimensions=self.dimensions,
            items=items,
        )

    async def close(self) -> None:
        return None

    def _embed(self, text: str) -> list[float]:
        normalized = text.casefold()
        vector = [0.0] * self.dimensions
        if any(token in normalized for token in ("todo", "task", "tasks", "reminder", "goal")):
            vector[0] += 1.0
        if any(token in normalized for token in ("roadmap", "plan")):
            vector[1] += 1.0
        if any(token in normalized for token in ("block", "blocked", "blocker", "latency")):
            vector[2] += 1.0
        if all(value == 0.0 for value in vector):
            vector[3] = 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector]


class FakeEntityAdjudicator:
    model_name = "fake-adjudicator"
    timezone_name = "UTC"

    def __init__(self, decision: EntityAdjudicationDecision) -> None:
        self.decision = decision
        self.requests = []

    async def adjudicate(self, request) -> EntityAdjudicationDecision:
        self.requests.append(request)
        return self.decision

    async def close(self) -> None:
        return None


class StaleHitEntityIndex:
    async def ensure_ready(self) -> None:
        return None

    async def sync_entity(self, entity) -> None:
        return None

    async def sync_entities(self, entities) -> None:
        return None

    async def delete_entities(self, entity_ids) -> None:
        return None

    async def search(self, query: str, *, entity_types=None, limit: int = 8):
        from cosmic_memory.graph.entity_index import EntitySimilarityHit

        return [
            EntitySimilarityHit(
                entity_id="ent_missing_from_graph",
                score=0.99,
                entity_type=EntityType.TASK,
                canonical_name="Missing task",
            )
        ]

    async def close(self) -> None:
        return None


def test_graph_store_merges_same_email_into_one_person():
    async def run():
        store = InMemoryGraphStore()
        first = await store.ingest_document(
            GraphDocument(
                memory_id="mem_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="person",
                        entity_type=EntityType.PERSON,
                        canonical_name="Nitin Agarwal",
                        identity_candidates=[
                            GraphIdentityCandidate(
                                key_type=IdentityKeyType.EMAIL,
                                raw_value="nxagarwal@ualr.edu",
                            )
                        ],
                        alias_values=["Dr. Nitin"],
                    )
                ],
            )
        )
        second = await store.ingest_document(
            GraphDocument(
                memory_id="mem_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="person",
                        entity_type=EntityType.PERSON,
                        canonical_name="Dr. Nitin",
                        identity_candidates=[
                            GraphIdentityCandidate(
                                key_type=IdentityKeyType.EMAIL,
                                raw_value="NXAGARWAL@UALR.EDU",
                            )
                        ],
                    )
                ],
            )
        )

        assert len(first.entity_ids) == 1
        assert second.entity_ids == first.entity_ids
        assert second.resolution_events[0].status == "exact_match"

    asyncio.run(run())


def test_graph_store_does_not_auto_merge_on_name_alias_only():
    async def run():
        store = InMemoryGraphStore()
        first = await store.ingest_document(
            GraphDocument(
                memory_id="mem_alias_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="person",
                        entity_type=EntityType.PERSON,
                        canonical_name="Dr. Nitin",
                    )
                ],
            )
        )
        second = await store.ingest_document(
            GraphDocument(
                memory_id="mem_alias_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="person",
                        entity_type=EntityType.PERSON,
                        canonical_name="Nitin",
                    )
                ],
            )
        )

        assert second.entity_ids[0] != first.entity_ids[0]
        assert second.resolution_events[0].status == "candidate_match"

    asyncio.run(run())


def test_passive_search_returns_one_hop_relation_from_email_seed():
    async def run():
        store = InMemoryGraphStore()
        await store.ingest_document(
            GraphDocument(
                memory_id="mem_rel_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="person",
                        entity_type=EntityType.PERSON,
                        canonical_name="Nitin Agarwal",
                        identity_candidates=[
                            GraphIdentityCandidate(
                                key_type=IdentityKeyType.EMAIL,
                                raw_value="nxagarwal@ualr.edu",
                            )
                        ],
                    ),
                    GraphDocumentEntity(
                        local_ref="project",
                        entity_type=EntityType.PROJECT,
                        canonical_name="Cosmic Memory",
                    ),
                ],
                relations=[
                    GraphDocumentRelation(
                        source_ref="person",
                        target_ref="project",
                        relation_type=RelationType.WORKS_ON,
                        fact="Nitin Agarwal works on Cosmic Memory.",
                    )
                ],
            )
        )

        result = await store.passive_search(
            build_query_frame("What project is nxagarwal@ualr.edu working on?")
        )

        assert result.relations
        assert result.relations[0].relation_type == RelationType.WORKS_ON
        assert "mem_rel_1" in result.supporting_memory_ids

    asyncio.run(run())


def test_graph_store_auto_merges_same_project_name_across_documents():
    async def run():
        store = InMemoryGraphStore()
        first = await store.ingest_document(
            GraphDocument(
                memory_id="mem_project_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="project",
                        entity_type=EntityType.PROJECT,
                        canonical_name="Cosmic Memory",
                    )
                ],
            )
        )
        second = await store.ingest_document(
            GraphDocument(
                memory_id="mem_project_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="project",
                        entity_type=EntityType.PROJECT,
                        canonical_name="Cosmic Memory",
                    )
                ],
            )
        )

        assert len(first.entity_ids) == 1
        assert second.entity_ids == first.entity_ids
        assert second.resolution_events[0].status == "exact_match"

    asyncio.run(run())


def test_graph_store_uses_entity_similarity_for_merge_candidates():
    async def run():
        store = InMemoryGraphStore(
            entity_index=InMemoryEntitySimilarityIndex(
                embedding_service=SemanticTestEmbeddingService()
            )
        )
        await store.ingest_document(
            GraphDocument(
                memory_id="mem_similarity_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="task",
                        entity_type=EntityType.TASK,
                        canonical_name="Roadmap task",
                        alias_values=["Roadmap action item"],
                    )
                ],
            )
        )
        result = await store.ingest_document(
            GraphDocument(
                memory_id="mem_similarity_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="todo",
                        entity_type=EntityType.REMINDER,
                        canonical_name="Roadmap todo",
                    )
                ],
            )
        )

        assert result.resolution_events[0].status == "candidate_match"
        assert result.resolution_events[0].candidates
        assert result.resolution_events[0].candidates[0].reason == "entity_similarity_candidate"

    asyncio.run(run())


def test_passive_search_can_seed_entities_from_entity_similarity():
    async def run():
        store = InMemoryGraphStore(
            entity_index=InMemoryEntitySimilarityIndex(
                embedding_service=SemanticTestEmbeddingService()
            )
        )
        await store.ingest_document(
            GraphDocument(
                memory_id="mem_similarity_rel",
                entities=[
                    GraphDocumentEntity(
                        local_ref="task",
                        entity_type=EntityType.TASK,
                        canonical_name="Roadmap task",
                    ),
                    GraphDocumentEntity(
                        local_ref="blocker",
                        entity_type=EntityType.TOPIC,
                        canonical_name="Embedding latency",
                    ),
                ],
                relations=[
                    GraphDocumentRelation(
                        source_ref="task",
                        target_ref="blocker",
                        relation_type=RelationType.BLOCKED_BY,
                        fact="Roadmap task is blocked by embedding latency.",
                    )
                ],
            )
        )

        result = await store.passive_search(build_query_frame("What is the roadmap todo blocked by?"))

        assert result.relations
        assert result.relations[0].relation_type == RelationType.BLOCKED_BY
        assert "mem_similarity_rel" in result.supporting_memory_ids

    asyncio.run(run())


def test_graph_store_can_use_internal_adjudicator_to_merge_similarity_candidates():
    async def run():
        adjudicator = FakeEntityAdjudicator(
            EntityAdjudicationDecision(
                decision="exact_match",
                chosen_entity_id="placeholder",
                confidence=0.94,
                rationale="todo and task clearly describe the same roadmap work item",
            )
        )
        store = InMemoryGraphStore(
            entity_index=InMemoryEntitySimilarityIndex(
                embedding_service=SemanticTestEmbeddingService()
            ),
            adjudicator=adjudicator,
        )
        first = await store.ingest_document(
            GraphDocument(
                memory_id="mem_adjudicate_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="task",
                        entity_type=EntityType.TASK,
                        canonical_name="Roadmap task",
                        alias_values=["Roadmap action item"],
                    )
                ],
                source_text="Roadmap task is the tracked work item for the plan.",
            )
        )
        existing_entity_id = first.entity_ids[0]
        adjudicator.decision.chosen_entity_id = existing_entity_id

        second = await store.ingest_document(
            GraphDocument(
                memory_id="mem_adjudicate_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="todo",
                        entity_type=EntityType.REMINDER,
                        canonical_name="Roadmap todo",
                    )
                ],
                source_text="Roadmap todo refers to the same work item already tracked in the roadmap task.",
            )
        )

        assert second.entity_ids == [existing_entity_id]
        assert second.resolution_events[0].status == "exact_match"
        assert adjudicator.requests
        assert adjudicator.requests[0].candidate_entities
        assert adjudicator.requests[0].candidate_entities[0].entity_id == existing_entity_id

    asyncio.run(run())


def test_graph_store_can_use_internal_adjudicator_to_create_new_entity():
    async def run():
        adjudicator = FakeEntityAdjudicator(
            EntityAdjudicationDecision(
                decision="created_new",
                confidence=0.82,
                rationale="the candidate entity is related but not the same thing",
            )
        )
        store = InMemoryGraphStore(
            entity_index=InMemoryEntitySimilarityIndex(
                embedding_service=SemanticTestEmbeddingService()
            ),
            adjudicator=adjudicator,
        )
        first = await store.ingest_document(
            GraphDocument(
                memory_id="mem_adjudicate_new_1",
                entities=[
                    GraphDocumentEntity(
                        local_ref="task",
                        entity_type=EntityType.TASK,
                        canonical_name="Roadmap task",
                    )
                ],
                source_text="Roadmap task is one tracked item.",
            )
        )
        second = await store.ingest_document(
            GraphDocument(
                memory_id="mem_adjudicate_new_2",
                entities=[
                    GraphDocumentEntity(
                        local_ref="todo",
                        entity_type=EntityType.REMINDER,
                        canonical_name="Roadmap todo",
                    )
                ],
                source_text="Roadmap todo is a separate reminder related to the task but not the same entity.",
            )
        )

        assert second.entity_ids[0] != first.entity_ids[0]
        assert second.resolution_events[0].status == "created_new"
        assert adjudicator.requests

    asyncio.run(run())


def test_passive_and_active_search_ignore_stale_entity_index_hits():
    async def run():
        store = InMemoryGraphStore(entity_index=StaleHitEntityIndex())
        await store.ingest_document(
            GraphDocument(
                memory_id="mem_stale_seed",
                entities=[
                    GraphDocumentEntity(
                        local_ref="task",
                        entity_type=EntityType.TASK,
                        canonical_name="Roadmap task",
                    )
                ],
                source_text="Roadmap task is the tracked work item for the plan.",
            )
        )

        passive = await store.passive_search(build_query_frame("What is the roadmap task?"))
        active = await store.traverse(build_query_frame("What is the roadmap task?"))

        assert passive.entities
        assert active.entities
        assert all(entity.entity_id != "ent_missing_from_graph" for entity in passive.entities)
        assert all(entity.entity_id != "ent_missing_from_graph" for entity in active.entities)

    asyncio.run(run())
