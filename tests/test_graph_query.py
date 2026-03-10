from cosmic_memory.graph.ontology import QueryIntent, RelationType
from cosmic_memory.graph.query import build_query_frame


def test_build_query_frame_detects_blocker_relation_and_current_state():
    frame = build_query_frame("What is blocking Cosmic Memory right now?")

    assert QueryIntent.TEMPORAL_LOOKUP in frame.intents
    assert RelationType.BLOCKED_BY in frame.allowed_relations
    assert frame.prefer_current_state is True


def test_build_query_frame_detects_decision_relation_keywords():
    frame = build_query_frame("What did we decide about Qdrant BM25 for Cosmic Memory?")

    assert QueryIntent.TASK_LOOKUP in frame.intents
    assert RelationType.DECIDED in frame.allowed_relations
