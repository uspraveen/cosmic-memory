import asyncio
from datetime import UTC, datetime

from cosmic_memory.control_surface import (
    CurrentStateRequest,
    MemoryBriefRequest,
    MemoryQueryPlanRequest,
    ResolveIdentityRequest,
    TemporalFactsRequest,
    build_memory_query_plan,
)
from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, WriteMemoryRequest
from cosmic_memory.graph import InMemoryGraphStore


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(
        source_kind="gateway",
        created_by="test",
        created_at=datetime(2026, 3, 10, 10, 0, tzinfo=UTC),
    )


def test_build_memory_query_plan_prefers_hybrid_for_current_relationship_query():
    plan = build_memory_query_plan(
        MemoryQueryPlanRequest(query="What is currently blocking Cosmic Memory and who works on it?"),
        graph_available=True,
    )

    assert plan.recommended_mode in {"hybrid", "active"}
    assert plan.include_schema_context is True
    assert "passive_search" in plan.tool_sequence
    assert "current_state" in plan.tool_sequence
    assert "memory_brief" in plan.tool_sequence


def test_agent_surface_methods_with_graph():
    async def run():
        service = InMemoryDevelopmentMemoryService(graph_store=InMemoryGraphStore())
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.TASK_SUMMARY,
                title="Current blocker",
                content="Cosmic Memory is currently blocked by embedding latency. Nitin works on Cosmic Memory right now.",
                metadata={
                    "graph_document": {
                        "memory_id": "placeholder",
                        "entities": [
                            {
                                "local_ref": "nitin",
                                "entity_type": "person",
                                "canonical_name": "Nitin Agarwal",
                                "identity_candidates": [
                                    {
                                        "key_type": "email",
                                        "raw_value": "nxagarwal@ualr.edu",
                                        "verified": False,
                                        "confidence": 1.0,
                                    }
                                ],
                                "alias_values": ["Dr. Nitin"],
                                "attributes": {},
                            },
                            {
                                "local_ref": "project",
                                "entity_type": "project",
                                "canonical_name": "Cosmic Memory",
                                "identity_candidates": [],
                                "alias_values": [],
                                "attributes": {},
                            },
                            {
                                "local_ref": "blocker",
                                "entity_type": "topic",
                                "canonical_name": "embedding latency",
                                "identity_candidates": [],
                                "alias_values": [],
                                "attributes": {},
                            },
                        ],
                        "relations": [
                            {
                                "source_ref": "project",
                                "target_ref": "blocker",
                                "relation_type": "blocked_by",
                                "fact": "Cosmic Memory is currently blocked by embedding latency.",
                                "confidence": 1.0,
                                "valid_at": "2026-03-10T10:00:00Z",
                                "invalid_at": None,
                                "expires_at": None,
                            },
                            {
                                "source_ref": "nitin",
                                "target_ref": "project",
                                "relation_type": "works_on",
                                "fact": "Nitin works on Cosmic Memory right now.",
                                "confidence": 1.0,
                                "valid_at": "2026-03-10T10:00:00Z",
                                "invalid_at": None,
                                "expires_at": None,
                            },
                        ],
                        "source_text": "Cosmic Memory is currently blocked by embedding latency. Nitin works on Cosmic Memory right now.",
                        "created_at": "2026-03-10T10:00:00Z",
                    }
                },
                provenance=_provenance(),
            )
        )

        schema = await service.get_schema_context()
        assert schema.graph_available is True
        assert any(tool.name == "memory_brief" for tool in schema.tools)

        resolved = await service.resolve_identity(
            ResolveIdentityRequest(value="NXAGARWAL@UALR.EDU", key_type="email")
        )
        assert resolved.status == "exact_match"
        assert resolved.entity is not None
        assert resolved.entity.name == "Nitin Agarwal"

        current = await service.get_current_state(
            CurrentStateRequest(query="What is blocking Cosmic Memory?")
        )
        assert current.facts
        assert "blocked_by" in {fact.relation_type for fact in current.facts}

        temporal = await service.get_temporal_facts(
            TemporalFactsRequest(query="When was Cosmic Memory blocked?")
        )
        assert temporal.facts
        assert temporal.facts[0].valid_at is not None

        brief = await service.build_memory_brief(
            MemoryBriefRequest(query="What is currently blocking Cosmic Memory?")
        )
        assert brief.findings
        assert record.memory_id in brief.supporting_memory_ids
        assert brief.plan.tool_sequence

    asyncio.run(run())
