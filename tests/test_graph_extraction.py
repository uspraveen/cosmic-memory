import asyncio
import sys
import types

from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    MemoryProvenance,
    MemoryRecord,
    WriteMemoryRequest,
)
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.extraction.normalize import normalize_extraction_result
from cosmic_memory.extraction.xai import XAIGraphExtractionService
from cosmic_memory.graph import (
    EntityType,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    RelationType,
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_normalize_extraction_result_deduplicates_entities_and_relations():
    record = MemoryRecord(
        memory_id="mem_extract_test",
        kind=MemoryKind.AGENT_NOTE,
        title="Duplicated extraction",
        content="Dr. Nitin works on Cosmic Memory. Nitin Agarwal works on Cosmic Memory.",
        provenance=provenance(),
    )

    result, report = normalize_extraction_result(
        GraphExtractionResult(
            should_extract=True,
            entities=[
                ExtractedGraphEntity(
                    local_ref="person_a",
                    entity_type=EntityType.PERSON,
                    canonical_name="Dr. Nitin",
                    identity_candidates=[
                        GraphIdentityCandidate(
                            key_type=IdentityKeyType.EMAIL,
                            raw_value="nxagarwal@ualr.edu",
                        )
                    ],
                    alias_values=["Nitin Agarwal"],
                ),
                ExtractedGraphEntity(
                    local_ref="person_b",
                    entity_type=EntityType.PERSON,
                    canonical_name="Nitin Agarwal",
                    identity_candidates=[
                        GraphIdentityCandidate(
                            key_type=IdentityKeyType.EMAIL,
                            raw_value="NXAGARWAL@UALR.EDU",
                        )
                    ],
                ),
                ExtractedGraphEntity(
                    local_ref="project_a",
                    entity_type=EntityType.PROJECT,
                    canonical_name="Cosmic Memory",
                ),
                ExtractedGraphEntity(
                    local_ref="project_b",
                    entity_type=EntityType.PROJECT,
                    canonical_name="Cosmic Memory",
                ),
            ],
            relations=[
                ExtractedGraphRelation(
                    source_ref="person_a",
                    target_ref="project_a",
                    relation_type=RelationType.WORKS_ON,
                    fact="Dr. Nitin works on Cosmic Memory.",
                ),
                ExtractedGraphRelation(
                    source_ref="person_b",
                    target_ref="project_b",
                    relation_type=RelationType.WORKS_ON,
                    fact="Dr. Nitin works on Cosmic Memory.",
                ),
            ],
        ),
        record=record,
    )

    assert result is not None
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert report.merged_entity_count == 2


def test_write_can_auto_extract_graph_document_and_ingest_it():
    class FakeExtractor:
        model_name = "fake-xai"

        async def extract(self, record):
            return GraphExtractionResult(
                should_extract=True,
                rationale="memory contains a grounded project relation",
                entities=[
                    ExtractedGraphEntity(
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
                    ExtractedGraphEntity(
                        local_ref="project",
                        entity_type=EntityType.PROJECT,
                        canonical_name="Cosmic Memory",
                    ),
                ],
                relations=[
                    ExtractedGraphRelation(
                        source_ref="person",
                        target_ref="project",
                        relation_type=RelationType.WORKS_ON,
                        fact="Nitin Agarwal works on Cosmic Memory.",
                    )
                ],
            )

        async def close(self):
            return None

    async def run():
        service = InMemoryDevelopmentMemoryService(
            graph_store=InMemoryGraphStore(),
            graph_extractor=FakeExtractor(),
        )
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title="Project relationship",
                content="Nitin Agarwal works on Cosmic Memory.",
                provenance=provenance(),
            )
        )
        assert "graph_document" in record.metadata
        assert record.metadata["graph_extraction"]["model"] == "fake-xai"

        response = await service.active_recall(
            ActiveRecallRequest(query="What project does nxagarwal@ualr.edu work on?")
        )

        assert response.relations
        assert response.relations[0].relation_type == "works_on"

    asyncio.run(run())


def test_xai_graph_extractor_builds_time_aware_prompt_and_parses_schema(monkeypatch):
    captured = {"messages": []}

    class FakeChat:
        def append(self, message):
            captured["messages"].append(message)

        def parse(self, schema):
            return (
                "unused-response",
                schema(
                    should_extract=True,
                    entities=[],
                    relations=[],
                    rationale="empty but valid parse",
                ),
            )

    class FakeChatAPI:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return FakeChat()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChatAPI()

        def close(self):
            return None

    fake_xai_sdk = types.ModuleType("xai_sdk")
    fake_xai_sdk_chat = types.ModuleType("xai_sdk.chat")
    fake_xai_sdk_chat.system = lambda text: {"role": "system", "content": text}
    fake_xai_sdk_chat.user = lambda text: {"role": "user", "content": text}
    fake_xai_sdk.Client = lambda api_key=None: FakeClient()

    monkeypatch.setitem(sys.modules, "xai_sdk", fake_xai_sdk)
    monkeypatch.setitem(sys.modules, "xai_sdk.chat", fake_xai_sdk_chat)

    async def run():
        extractor = XAIGraphExtractionService(
            client=FakeClient(),
            timezone_name="America/Chicago",
        )
        result = await extractor.extract(
            MemoryRecord(
                memory_id="mem_xai_prompt",
                kind=MemoryKind.TASK_SUMMARY,
                title="Blocker",
                content="Today Cosmic Memory is blocked by embedding latency.",
                provenance=provenance(),
            )
        )

        assert result is not None
        assert captured["kwargs"]["model"] == "grok-4-1-fast-reasoning"
        user_message = captured["messages"][1]["content"]
        assert "Current UTC time:" in user_message
        assert "Current local time (America/Chicago):" in user_message
        assert "Memory created_at:" in user_message
        assert "Today Cosmic Memory is blocked by embedding latency." in user_message

    asyncio.run(run())
