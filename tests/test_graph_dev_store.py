import asyncio

from cosmic_memory.graph import (
    EntityType,
    GraphDocument,
    GraphDocumentEntity,
    GraphDocumentRelation,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    RelationType,
    build_query_frame,
)


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
