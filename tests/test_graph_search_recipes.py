from datetime import datetime, timedelta, timezone

from cosmic_memory.graph import (
    EntityType,
    GraphEntityNode,
    GraphEpisode,
    GraphQueryFrame,
    GraphRelationEdge,
    GraphSearchResult,
    QueryIntent,
    RelationType,
    apply_graph_search_recipe,
)


def test_active_current_state_recipe_prefers_active_direct_relation():
    now = datetime.now(timezone.utc)
    active_episode = GraphEpisode(
        episode_id="ep_active",
        memory_id="mem_active",
        source_type="agent_note",
        created_at=now,
        extracted_at=now,
        extraction_confidence=0.92,
        source_excerpt="Cosmic Memory is blocked by Neo4j provisioning.",
        produced_relation_ids=["rel_active"],
    )
    stale_episode = GraphEpisode(
        episode_id="ep_stale",
        memory_id="mem_stale",
        source_type="session_summary",
        created_at=now - timedelta(days=7),
        extracted_at=now - timedelta(days=7),
        extraction_confidence=0.86,
        source_excerpt="Cosmic Memory used to be blocked by embedding latency.",
        produced_relation_ids=["rel_stale"],
        invalidated_relation_ids=["rel_stale"],
    )
    project = GraphEntityNode(
        entity_id="ent_project",
        entity_type=EntityType.PROJECT,
        canonical_name="Cosmic Memory",
        memory_ids=["mem_active", "mem_stale"],
    )
    blocker_active = GraphEntityNode(
        entity_id="ent_blocker_active",
        entity_type=EntityType.TOPIC,
        canonical_name="Neo4j provisioning",
        memory_ids=["mem_active"],
    )
    blocker_stale = GraphEntityNode(
        entity_id="ent_blocker_stale",
        entity_type=EntityType.TOPIC,
        canonical_name="Embedding latency",
        memory_ids=["mem_stale"],
    )
    active_relation = GraphRelationEdge(
        relation_id="rel_active",
        relation_type=RelationType.BLOCKED_BY,
        source_entity_id=project.entity_id,
        target_entity_id=blocker_active.entity_id,
        fact="Cosmic Memory is blocked by Neo4j provisioning.",
        memory_ids=["mem_active"],
        episode_ids=[active_episode.episode_id],
        created_at=now,
        updated_at=now,
    )
    stale_relation = GraphRelationEdge(
        relation_id="rel_stale",
        relation_type=RelationType.BLOCKED_BY,
        source_entity_id=project.entity_id,
        target_entity_id=blocker_stale.entity_id,
        fact="Cosmic Memory is blocked by embedding latency.",
        memory_ids=["mem_stale"],
        episode_ids=[stale_episode.episode_id],
        created_at=now - timedelta(days=7),
        updated_at=now - timedelta(days=1),
        invalidated_by_episode_id=active_episode.episode_id,
    )
    graph_result = GraphSearchResult(
        entities=[project, blocker_active, blocker_stale],
        relations=[stale_relation, active_relation],
        episodes=[stale_episode, active_episode],
        seed_entity_ids=[project.entity_id],
        relation_distances={"rel_active": 1, "rel_stale": 2},
        supporting_memory_ids=["mem_stale", "mem_active"],
        search_plan=["graph traverse"],
    )
    query_frame = GraphQueryFrame(
        query="What is blocking Cosmic Memory right now?",
        intents=[QueryIntent.TASK_LOOKUP],
        entity_terms=["cosmic", "memory", "blocking"],
        allowed_relations=[RelationType.BLOCKED_BY],
        prefer_current_state=True,
    )

    applied = apply_graph_search_recipe(
        graph_result=graph_result,
        query_frame=query_frame,
        mode="active",
        max_results=2,
    )

    assert applied.recipe_name == "active_current_state_rrf_mmr"
    assert [relation.relation_id for relation in applied.graph_result.relations][:2] == [
        "rel_active",
        "rel_stale",
    ]
    assert applied.graph_result.supporting_memory_ids[0] == "mem_active"
    assert applied.memory_boosts["mem_active"] > applied.memory_boosts["mem_stale"]


def test_active_recipe_mmr_diversifies_near_duplicate_relations():
    now = datetime.now(timezone.utc)
    episode = GraphEpisode(
        episode_id="ep_1",
        memory_id="mem_1",
        source_type="agent_note",
        created_at=now,
        extracted_at=now,
        extraction_confidence=0.9,
    )
    project = GraphEntityNode(
        entity_id="ent_project",
        entity_type=EntityType.PROJECT,
        canonical_name="Cosmic Memory",
        memory_ids=["mem_1", "mem_2", "mem_3"],
    )
    blocker_a = GraphEntityNode(
        entity_id="ent_blocker_a",
        entity_type=EntityType.TOPIC,
        canonical_name="Embedding latency",
        memory_ids=["mem_1", "mem_2"],
    )
    blocker_b = GraphEntityNode(
        entity_id="ent_blocker_b",
        entity_type=EntityType.TOPIC,
        canonical_name="Neo4j provisioning",
        memory_ids=["mem_3"],
    )
    relation_a = GraphRelationEdge(
        relation_id="rel_a",
        relation_type=RelationType.BLOCKED_BY,
        source_entity_id=project.entity_id,
        target_entity_id=blocker_a.entity_id,
        fact="Cosmic Memory is blocked by embedding latency.",
        memory_ids=["mem_1"],
        episode_ids=[episode.episode_id],
        created_at=now,
        updated_at=now,
    )
    relation_b = GraphRelationEdge(
        relation_id="rel_b",
        relation_type=RelationType.BLOCKED_BY,
        source_entity_id=project.entity_id,
        target_entity_id=blocker_a.entity_id,
        fact="Cosmic Memory remains blocked by embedding latency issue.",
        memory_ids=["mem_2"],
        episode_ids=[episode.episode_id],
        created_at=now,
        updated_at=now,
    )
    relation_c = GraphRelationEdge(
        relation_id="rel_c",
        relation_type=RelationType.BLOCKED_BY,
        source_entity_id=project.entity_id,
        target_entity_id=blocker_b.entity_id,
        fact="Cosmic Memory is blocked by Neo4j provisioning.",
        memory_ids=["mem_3"],
        episode_ids=[episode.episode_id],
        created_at=now,
        updated_at=now,
    )
    graph_result = GraphSearchResult(
        entities=[project, blocker_a, blocker_b],
        relations=[relation_a, relation_b, relation_c],
        episodes=[episode],
        seed_entity_ids=[project.entity_id],
        relation_distances={"rel_a": 1, "rel_b": 1, "rel_c": 1},
        supporting_memory_ids=["mem_1", "mem_2", "mem_3"],
        search_plan=["graph traverse"],
    )
    query_frame = GraphQueryFrame(
        query="What blockers does Cosmic Memory have?",
        intents=[QueryIntent.TASK_LOOKUP],
        entity_terms=["blockers", "cosmic", "memory"],
        allowed_relations=[RelationType.BLOCKED_BY],
        prefer_current_state=False,
    )

    applied = apply_graph_search_recipe(
        graph_result=graph_result,
        query_frame=query_frame,
        mode="active",
        max_results=2,
    )

    selected_ids = [relation.relation_id for relation in applied.graph_result.relations]
    assert "rel_a" in selected_ids
    assert "rel_c" in selected_ids
    assert "rel_b" not in selected_ids
