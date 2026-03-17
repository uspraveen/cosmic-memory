import asyncio
import shutil
import sys
import types
from pathlib import Path

from datetime import UTC, datetime

from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    GraphSyncRequest,
    MemoryProvenance,
    MemoryRecord,
    WriteMemoryRequest,
)
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.extraction.deterministic import DeterministicGraphExtractionService
from cosmic_memory.extraction.normalize import normalize_extraction_result
from cosmic_memory.extraction.xai import XAIGraphExtractionService
from cosmic_memory.graph import (
    EntityType,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    Neo4jGraphStore,
    RelationType,
)
from cosmic_memory.graph.models import GraphDocument, GraphEpisode


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


def test_deterministic_extractor_can_capture_primary_user_preference():
    async def run():
        service = InMemoryDevelopmentMemoryService(
            graph_store=InMemoryGraphStore(),
            graph_extractor=DeterministicGraphExtractionService(
                primary_user_display_name="Praveen"
            ),
        )
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Favorite hero",
                content="User loves iron man",
                provenance=provenance(),
            )
        )
        document = record.metadata["graph_document"]
        assert document["entities"][0]["canonical_name"] == "Praveen"
        assert document["relations"][0]["relation_type"] == "prefers"

        response = await service.active_recall(
            ActiveRecallRequest(query="What does Praveen prefer?")
        )

        assert response.relations
        assert response.relations[0].relation_type == "prefers"

    asyncio.run(run())


def test_deterministic_extractor_can_capture_primary_user_education_relation():
    async def run():
        service = InMemoryDevelopmentMemoryService(
            graph_store=InMemoryGraphStore(),
            graph_extractor=DeterministicGraphExtractionService(
                primary_user_display_name="Praveen"
            ),
        )
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Education",
                content="I graduated from University of Pennsylvania (UPenn) in 2022.",
                provenance=provenance(),
            )
        )

        document = record.metadata["graph_document"]
        institution = next(
            entity for entity in document["entities"] if entity["entity_type"] == "organization"
        )
        assert institution["canonical_name"] == "University of Pennsylvania"
        assert "UPenn" in institution["alias_values"]
        assert document["relations"][0]["relation_type"] == "graduated_from"

        response = await service.active_recall(
            ActiveRecallRequest(query="Where did Praveen graduate from?")
        )

        assert response.relations
        assert response.relations[0].relation_type == "graduated_from"

    asyncio.run(run())


def test_sync_graph_can_backfill_deterministic_documents_without_llm():
    async def run():
        seed_service = InMemoryDevelopmentMemoryService()
        record = await seed_service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Favorite hero",
                content="User loves iron man",
                provenance=provenance(),
            )
        )

        from cosmic_memory.filesystem_service import FilesystemMemoryService

        temp_dir = Path(".manual_graph_sync_backfill_case")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            filesystem_service = FilesystemMemoryService(
                temp_dir,
                graph_store=InMemoryGraphStore(),
                graph_extractor=DeterministicGraphExtractionService(
                    primary_user_display_name="Praveen"
                ),
            )
            write_result = filesystem_service.record_store.write(record)
            filesystem_service.registry.upsert(
                record,
                write_result.path,
                write_result.content_hash,
            )

            sync = await filesystem_service.sync_graph()

            assert sync.enabled is True
            assert sync.graph_upserts == 1
            assert sync.status.ingested_memory_count == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run())


def test_normalize_extraction_result_rewrites_legacy_part_of_education_relation():
    record = MemoryRecord(
        memory_id="mem_education_rewrite",
        kind=MemoryKind.USER_DATA,
        title="Education",
        content="Praveen graduated from University of Pennsylvania (UPenn) in 2022.",
        provenance=provenance(),
    )

    result, report = normalize_extraction_result(
        GraphExtractionResult(
            should_extract=True,
            entities=[
                ExtractedGraphEntity(
                    local_ref="user",
                    entity_type=EntityType.PERSON,
                    canonical_name="Praveen",
                ),
                ExtractedGraphEntity(
                    local_ref="school",
                    entity_type=EntityType.ORGANIZATION,
                    canonical_name="University of Pennsylvania (UPenn)",
                ),
            ],
            relations=[
                ExtractedGraphRelation(
                    source_ref="user",
                    target_ref="school",
                    relation_type=RelationType.PART_OF,
                    fact="Praveen graduated from University of Pennsylvania (UPenn) in 2022.",
                )
            ],
        ),
        record=record,
    )

    assert result is not None
    assert report.rewritten_relation_count == 1
    assert result.entities[1].canonical_name == "University of Pennsylvania"
    assert "UPenn" in result.entities[1].alias_values
    assert result.relations[0].relation_type == RelationType.GRADUATED_FROM


def test_graph_store_merges_semantically_equivalent_graduation_facts():
    class SequencedExtractor:
        model_name = "sequenced-education"

        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, _record):
            self.calls += 1
            fact = (
                "Praveen graduated from UPenn in 2022."
                if self.calls == 1
                else "Praveen graduated from the University of Pennsylvania (UPenn) in 2022."
            )
            return GraphExtractionResult(
                should_extract=True,
                rationale="education_relation",
                entities=[
                    ExtractedGraphEntity(
                        local_ref="user",
                        entity_type=EntityType.PERSON,
                        canonical_name="Praveen",
                        identity_candidates=[
                            GraphIdentityCandidate(
                                key_type=IdentityKeyType.EXTERNAL_ACCOUNT,
                                provider="cosmic",
                                raw_value="primary_user",
                                verified=True,
                            )
                        ],
                    ),
                    ExtractedGraphEntity(
                        local_ref="school",
                        entity_type=EntityType.ORGANIZATION,
                        canonical_name="University of Pennsylvania (UPenn)",
                    ),
                ],
                relations=[
                    ExtractedGraphRelation(
                        source_ref="user",
                        target_ref="school",
                        relation_type=RelationType.GRADUATED_FROM,
                        fact=fact,
                    )
                ],
            )

        async def close(self):
            return None

    async def run():
        service = InMemoryDevelopmentMemoryService(
            graph_store=InMemoryGraphStore(),
            graph_extractor=SequencedExtractor(),
        )
        for title in ("Education one", "Education two"):
            await service.write(
                WriteMemoryRequest(
                    kind=MemoryKind.USER_DATA,
                    title=title,
                    content=title,
                    provenance=provenance(),
                )
            )

        response = await service.active_recall(
            ActiveRecallRequest(query="Where did Praveen graduate from?")
        )

        graduation_relations = [
            relation for relation in response.relations if relation.relation_type == "graduated_from"
        ]
        assert len(graduation_relations) == 1
        assert len(graduation_relations[0].memory_ids) == 2

    asyncio.run(run())


def test_sync_graph_can_persist_backfilled_graph_documents():
    async def run():
        from cosmic_memory.filesystem_service import FilesystemMemoryService

        temp_dir = Path(".manual_graph_sync_persist_case")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            service = FilesystemMemoryService(
                temp_dir,
                graph_store=InMemoryGraphStore(),
                graph_extractor=DeterministicGraphExtractionService(
                    primary_user_display_name="Praveen"
                ),
            )
            record = MemoryRecord(
                kind=MemoryKind.CORE_FACT,
                title="Favorite hero",
                content="User loves iron man",
                provenance=provenance(),
            )
            write_result = service.record_store.write(record)
            service.registry.upsert(record, write_result.path, write_result.content_hash)

            sync = await service.sync_graph(
                GraphSyncRequest(
                    persist_graph_documents=True,
                    only_missing_graph_documents=True,
                )
            )
            reloaded = service.record_store.read(write_result.path)

            assert sync.enabled is True
            assert sync.persisted_graph_document_writes == 1
            assert sync.status.persisted_graph_document_count == 1
            assert reloaded.metadata["graph_document"]["entities"][0]["canonical_name"] == "Praveen"
            assert reloaded.metadata["graph_extraction"]["mode"] == "deterministic"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run())


def test_sync_graph_can_rewrite_persisted_legacy_education_relations_without_llm():
    async def run():
        from cosmic_memory.filesystem_service import FilesystemMemoryService

        temp_dir = Path(".manual_graph_sync_rewrite_case")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            service = FilesystemMemoryService(
                temp_dir,
                graph_store=InMemoryGraphStore(),
                graph_extractor=DeterministicGraphExtractionService(
                    primary_user_display_name="Praveen"
                ),
            )
            record = MemoryRecord(
                kind=MemoryKind.USER_DATA,
                title="Education",
                content="Praveen graduated from University of Pennsylvania (UPenn) in 2022.",
                metadata={
                    "graph_document": {
                        "memory_id": "ignored",
                        "entities": [
                            {
                                "local_ref": "user",
                                "entity_type": "person",
                                "canonical_name": "Praveen",
                            },
                            {
                                "local_ref": "school",
                                "entity_type": "organization",
                                "canonical_name": "University of Pennsylvania (UPenn)",
                            },
                        ],
                        "relations": [
                            {
                                "source_ref": "user",
                                "target_ref": "school",
                                "relation_type": "part_of",
                                "fact": "Praveen graduated from University of Pennsylvania (UPenn) in 2022.",
                            }
                        ],
                    }
                },
                provenance=provenance(),
            )
            write_result = service.record_store.write(record)
            service.registry.upsert(record, write_result.path, write_result.content_hash)

            sync = await service.sync_graph(
                GraphSyncRequest(
                    persist_graph_documents=True,
                    only_missing_graph_documents=False,
                )
            )
            reloaded = service.record_store.read(write_result.path)

            assert sync.enabled is True
            assert sync.persisted_graph_document_writes == 1
            assert reloaded.metadata["graph_document"]["relations"][0]["relation_type"] == "graduated_from"
            assert (
                reloaded.metadata["graph_document"]["entities"][1]["canonical_name"]
                == "University of Pennsylvania"
            )
            assert "UPenn" in reloaded.metadata["graph_document"]["entities"][1]["alias_values"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run())


def test_sync_graph_can_limit_llm_backfill_to_missing_documents():
    class FakeLLMExtractor:
        model_name = "fake-llm"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def extract(self, record):
            self.calls.append(record.memory_id)
            return GraphExtractionResult(
                should_extract=True,
                rationale="capture stable preference relation",
                entities=[
                    ExtractedGraphEntity(
                        local_ref="user",
                        entity_type=EntityType.PERSON,
                        canonical_name="Praveen",
                    ),
                    ExtractedGraphEntity(
                        local_ref="topic",
                        entity_type=EntityType.TOPIC,
                        canonical_name="Iron Man",
                    ),
                ],
                relations=[
                    ExtractedGraphRelation(
                        source_ref="user",
                        target_ref="topic",
                        relation_type=RelationType.PREFERS,
                        fact=record.content,
                    )
                ],
            )

    async def run():
        from cosmic_memory.filesystem_service import FilesystemMemoryService

        temp_dir = Path(".manual_graph_sync_llm_limit_case")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            llm_extractor = FakeLLMExtractor()
            service = FilesystemMemoryService(
                temp_dir,
                graph_store=InMemoryGraphStore(),
                graph_extractor=None,
                graph_llm_extractor=llm_extractor,
            )
            seeded_record = MemoryRecord(
                kind=MemoryKind.CORE_FACT,
                title="Already prepared",
                content="User prefers concise answers",
                provenance=provenance(),
                metadata={
                    "graph_document": {
                        "memory_id": "ignored",
                        "entities": [
                            {
                                "local_ref": "user",
                                "entity_type": "person",
                                "canonical_name": "Praveen",
                            },
                            {
                                "local_ref": "topic",
                                "entity_type": "topic",
                                "canonical_name": "concise answers",
                            },
                        ],
                        "relations": [
                            {
                                "source_ref": "user",
                                "target_ref": "topic",
                                "relation_type": "prefers",
                                "fact": "User prefers concise answers",
                            }
                        ],
                    }
                },
            )
            missing_record = MemoryRecord(
                kind=MemoryKind.CORE_FACT,
                title="Needs backfill",
                content="User loves iron man",
                provenance=provenance(),
            )
            for record in (seeded_record, missing_record):
                write_result = service.record_store.write(record)
                service.registry.upsert(record, write_result.path, write_result.content_hash)

            sync = await service.sync_graph(
                GraphSyncRequest(
                    allow_llm=True,
                    persist_graph_documents=True,
                    only_missing_graph_documents=True,
                    max_records=1,
                )
            )
            reloaded = await service.get(missing_record.memory_id)

            assert sync.enabled is True
            assert sync.target_memory_count == 1
            assert sync.persisted_graph_document_writes == 1
            assert llm_extractor.calls == [missing_record.memory_id]
            assert reloaded is not None
            assert reloaded.metadata["graph_extraction"]["mode"] == "llm"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run())


def test_sync_graph_continues_after_record_failure_and_reports_failed_ids():
    class FlakyLLMExtractor:
        model_name = "fake-llm"

        async def extract(self, record):
            if "fails" in record.title.lower():
                raise RuntimeError("synthetic extractor failure")
            return GraphExtractionResult(
                should_extract=True,
                rationale="capture stable preference relation",
                entities=[
                    ExtractedGraphEntity(
                        local_ref="user",
                        entity_type=EntityType.PERSON,
                        canonical_name="Praveen",
                    ),
                    ExtractedGraphEntity(
                        local_ref="topic",
                        entity_type=EntityType.TOPIC,
                        canonical_name="Iron Man",
                    ),
                ],
                relations=[
                    ExtractedGraphRelation(
                        source_ref="user",
                        target_ref="topic",
                        relation_type=RelationType.PREFERS,
                        fact=record.content,
                    )
                ],
            )

    async def run():
        from cosmic_memory.filesystem_service import FilesystemMemoryService

        temp_dir = Path(".manual_graph_sync_failure_case")
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            service = FilesystemMemoryService(
                temp_dir,
                graph_store=InMemoryGraphStore(),
                graph_extractor=None,
                graph_llm_extractor=FlakyLLMExtractor(),
            )
            good_record = MemoryRecord(
                kind=MemoryKind.CORE_FACT,
                title="Works",
                content="User loves iron man",
                provenance=provenance(),
            )
            bad_record = MemoryRecord(
                kind=MemoryKind.CORE_FACT,
                title="Fails extractor",
                content="User likes thor",
                provenance=provenance(),
            )
            for record in (good_record, bad_record):
                write_result = service.record_store.write(record)
                service.registry.upsert(record, write_result.path, write_result.content_hash)

            sync = await service.sync_graph(
                GraphSyncRequest(
                    allow_llm=True,
                    persist_graph_documents=True,
                    only_missing_graph_documents=True,
                )
            )

            assert sync.enabled is True
            assert sync.graph_upserts == 1
            assert sync.failed_memory_count == 1
            assert sync.failed_memory_ids == [bad_record.memory_id]
            assert sync.status.ingested_memory_count == 1
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    asyncio.run(run())


def test_neo4j_pending_relation_uses_resolved_entity_names_without_requery():
    async def run():
        store = object.__new__(Neo4jGraphStore)
        persisted_relations = []
        linked_relation_ids = []

        async def _get_relation(_session, _relation_id):
            return None

        async def _find_facts(_query):
            return []

        async def _maybe_invalidate_relations(**_kwargs):
            return "create", None, []

        async def _persist_relation(_session, relation):
            persisted_relations.append(relation)

        async def _link_episode_to_relation(_session, *, relation_id, **_kwargs):
            linked_relation_ids.append(relation_id)

        async def _unexpected_get_entity(_entity_id):
            raise AssertionError("new relation upsert should not re-query pending entities")

        store._get_relation = _get_relation
        store.find_facts = _find_facts
        store._maybe_invalidate_relations = _maybe_invalidate_relations
        store._persist_relation = _persist_relation
        store._link_episode_to_relation = _link_episode_to_relation
        store.get_entity = _unexpected_get_entity

        document = GraphDocument(memory_id="mem_graph_relation")
        episode = GraphEpisode(
            episode_id="ep_graph_relation",
            memory_id=document.memory_id,
            source_type="memory_record",
        )

        edge, invalidated = await Neo4jGraphStore._upsert_relation(
            store,
            object(),
            document=document,
            episode=episode,
            memory_id=document.memory_id,
            source_entity_id="entity_praveen",
            target_entity_id="entity_iron_man",
            source_entity_name="Praveen",
            target_entity_name="Iron Man",
            relation_type=RelationType.PREFERS,
            fact="User loves iron man",
            confidence=1.0,
            valid_at=None,
            invalid_at=None,
            expires_at=None,
        )

        assert edge is not None
        assert edge.relation_type == RelationType.PREFERS
        assert invalidated == []
        assert persisted_relations
        assert linked_relation_ids == [edge.relation_id]

    asyncio.run(run())


def test_normalize_extraction_result_backfills_current_relation_valid_at():
    anchor = datetime(2026, 3, 10, 9, 30, tzinfo=UTC)
    record = MemoryRecord(
        memory_id="mem_temporal_backfill",
        kind=MemoryKind.TASK_SUMMARY,
        title="Current blocker",
        content="Today Cosmic Memory is currently blocked by embedding latency.",
        provenance=MemoryProvenance(
            source_kind="gateway",
            created_by="test",
            created_at=anchor,
        ),
        created_at=anchor,
        updated_at=anchor,
    )

    result, _report = normalize_extraction_result(
        GraphExtractionResult(
            should_extract=True,
            entities=[
                ExtractedGraphEntity(
                    local_ref="project",
                    entity_type=EntityType.PROJECT,
                    canonical_name="Cosmic Memory",
                ),
                ExtractedGraphEntity(
                    local_ref="blocker",
                    entity_type=EntityType.TOPIC,
                    canonical_name="embedding latency",
                ),
            ],
            relations=[
                ExtractedGraphRelation(
                    source_ref="project",
                    target_ref="blocker",
                    relation_type=RelationType.BLOCKED_BY,
                    fact="Cosmic Memory is currently blocked by embedding latency.",
                )
            ],
        ),
        record=record,
    )

    assert result is not None
    assert result.relations[0].valid_at == anchor


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
            primary_user_display_name="Praveen",
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
        system_message = captured["messages"][0]["content"]
        user_message = captured["messages"][1]["content"]
        assert "set valid_at to the provided Provenance created_at anchor" in system_message
        assert "Current UTC time:" in user_message
        assert "Current local time (America/Chicago):" in user_message
        assert "Memory created_at:" in user_message
        assert "Provenance created_at:" in user_message
        assert "primary_user_display_name: Praveen" in user_message
        assert "Today Cosmic Memory is blocked by embedding latency." in user_message
        assert "graduated_from" in user_message
        assert "attended" in user_message

    asyncio.run(run())
