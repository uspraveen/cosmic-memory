from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import RecallItem
from cosmic_memory.retrieval import rerank_passive_items, select_passive_items


def test_select_passive_items_prefers_budget_fit_over_oversized_anchor():
    items = [
        RecallItem(
            memory_id="mem_large",
            kind=MemoryKind.USER_DATA,
            title="Large import",
            content="x " * 90,
            score=1.20,
            tags=["imported"],
            token_count=90,
        ),
        RecallItem(
            memory_id="mem_small_a",
            kind=MemoryKind.AGENT_NOTE,
            title="Useful note",
            content="small relevant note",
            score=1.05,
            tags=["learning"],
            token_count=10,
        ),
        RecallItem(
            memory_id="mem_small_b",
            kind=MemoryKind.SESSION_SUMMARY,
            title="Useful summary",
            content="small relevant summary",
            score=0.95,
            tags=["current"],
            token_count=10,
        ),
    ]

    selected, total = select_passive_items(
        items,
        max_results=3,
        token_budget=25,
    )

    assert [item.memory_id for item in selected] == ["mem_small_a", "mem_small_b"]
    assert total == 20


def test_rerank_passive_items_boosts_exact_email_hits():
    items = [
        RecallItem(
            memory_id="mem_alias",
            kind=MemoryKind.USER_DATA,
            title="Profile",
            content="Dr. Nitin works on graph retrieval.",
            score=0.80,
            tags=["profile"],
            token_count=8,
        ),
        RecallItem(
            memory_id="mem_email",
            kind=MemoryKind.USER_DATA,
            title="Directory entry",
            content="nxagarwal@ualr.edu works on graph retrieval.",
            score=0.80,
            tags=["profile"],
            token_count=7,
        ),
    ]

    reranked = rerank_passive_items(
        items,
        query="What does nxagarwal@ualr.edu work on?",
    )

    assert [item.memory_id for item in reranked][:2] == ["mem_email", "mem_alias"]
