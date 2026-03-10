import asyncio

from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    MemoryProvenance,
    PassiveRecallRequest,
    SupersedeMemoryRequest,
    WriteMemoryRequest,
)
from cosmic_memory.graph import (
    EntityType,
    GraphDocument,
    GraphDocumentEntity,
    GraphDocumentRelation,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    RelationType,
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_active_recall_prefers_graph_traversal_when_available():
    async def run():
        graph_store = InMemoryGraphStore()
        service = InMemoryDevelopmentMemoryService(graph_store=graph_store)

        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title="Project relationship",
                content="Nitin Agarwal works on Cosmic Memory.",
                provenance=provenance(),
                metadata={
                    "graph_document": GraphDocument(
                        memory_id="ignored",
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
                },
            )
        )

        result = await service.active_recall(
            ActiveRecallRequest(
                query="What project does nxagarwal@ualr.edu work on?",
                include_diagnostics=True,
            )
        )

        assert result.items
        assert result.items[0].memory_id == record.memory_id
        assert result.entities
        assert result.relations
        assert result.relations[0].relation_type == RelationType.WORKS_ON.value
        assert result.diagnostics is not None
        assert result.diagnostics.flags["graph_used"] is True
        assert result.diagnostics.timings_ms["graph_traverse_ms"] >= 0

    asyncio.run(run())


def test_passive_recall_can_be_boosted_by_graph_supporting_memory():
    async def run():
        graph_store = InMemoryGraphStore()
        service = InMemoryDevelopmentMemoryService(graph_store=graph_store)

        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Research profile",
                content="Nitin Agarwal works on Cosmic Memory and graph retrieval.",
                provenance=provenance(),
                metadata={
                    "graph_document": {
                        "memory_id": "ignored",
                        "entities": [
                            {
                                "local_ref": "person",
                                "entity_type": EntityType.PERSON.value,
                                "canonical_name": "Nitin Agarwal",
                                "identity_candidates": [
                                    {
                                        "key_type": IdentityKeyType.EMAIL.value,
                                        "raw_value": "nxagarwal@ualr.edu",
                                    }
                                ],
                            },
                            {
                                "local_ref": "project",
                                "entity_type": EntityType.PROJECT.value,
                                "canonical_name": "Cosmic Memory",
                            },
                        ],
                        "relations": [
                            {
                                "source_ref": "person",
                                "target_ref": "project",
                                "relation_type": RelationType.WORKS_ON.value,
                                "fact": "Nitin Agarwal works on Cosmic Memory.",
                            }
                        ],
                    }
                },
            )
        )

        result = await service.passive_recall(
            PassiveRecallRequest(
                query="What project is nxagarwal@ualr.edu working on?",
                max_results=4,
                token_budget=80,
                include_diagnostics=True,
            )
        )

        assert result.items
        assert result.items[0].memory_id == record.memory_id
        assert result.diagnostics is not None
        assert result.diagnostics.flags["graph_assist_requested"] is True
        assert result.diagnostics.flags["graph_assist_used"] is True
        assert result.diagnostics.timings_ms["graph_wait_ms"] >= 0

    asyncio.run(run())


def test_supersede_removes_old_graph_contribution():
    async def run():
        graph_store = InMemoryGraphStore()
        service = InMemoryDevelopmentMemoryService(graph_store=graph_store)

        original = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title="Old relation",
                content="Nitin works on Old Project.",
                provenance=provenance(),
                metadata={
                    "graph_document": {
                        "memory_id": "ignored",
                        "entities": [
                            {
                                "local_ref": "person",
                                "entity_type": EntityType.PERSON.value,
                                "canonical_name": "Nitin Agarwal",
                                "identity_candidates": [
                                    {
                                        "key_type": IdentityKeyType.EMAIL.value,
                                        "raw_value": "nxagarwal@ualr.edu",
                                    }
                                ],
                            },
                            {
                                "local_ref": "project",
                                "entity_type": EntityType.PROJECT.value,
                                "canonical_name": "Old Project",
                            },
                        ],
                        "relations": [
                            {
                                "source_ref": "person",
                                "target_ref": "project",
                                "relation_type": RelationType.WORKS_ON.value,
                                "fact": "Nitin works on Old Project.",
                            }
                        ],
                    }
                },
            )
        )

        await service.supersede(
            original.memory_id,
            SupersedeMemoryRequest(
                replacement=WriteMemoryRequest(
                    kind=MemoryKind.AGENT_NOTE,
                    title="New relation",
                    content="Nitin works on Cosmic Memory.",
                    provenance=provenance(),
                    metadata={
                        "graph_document": {
                            "memory_id": "ignored",
                            "entities": [
                                {
                                    "local_ref": "person",
                                    "entity_type": EntityType.PERSON.value,
                                    "canonical_name": "Nitin Agarwal",
                                    "identity_candidates": [
                                        {
                                            "key_type": IdentityKeyType.EMAIL.value,
                                            "raw_value": "nxagarwal@ualr.edu",
                                        }
                                    ],
                                },
                                {
                                    "local_ref": "project",
                                    "entity_type": EntityType.PROJECT.value,
                                    "canonical_name": "Cosmic Memory",
                                },
                            ],
                            "relations": [
                                {
                                    "source_ref": "person",
                                    "target_ref": "project",
                                    "relation_type": RelationType.WORKS_ON.value,
                                    "fact": "Nitin works on Cosmic Memory.",
                                }
                            ],
                        }
                    },
                )
            ),
        )

        result = await service.active_recall(
            ActiveRecallRequest(query="What project does nxagarwal@ualr.edu work on?")
        )

        assert result.relations
        assert all("Old Project" not in relation.fact for relation in result.relations)
        assert any("Cosmic Memory" in relation.fact for relation in result.relations)

    asyncio.run(run())
