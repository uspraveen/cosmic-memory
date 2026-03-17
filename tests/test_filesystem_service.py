import asyncio

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    MemoryRecord,
    MemoryProvenance,
    PassiveRecallRequest,
    SupersedeMemoryRequest,
    WriteMemoryRequest,
)
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import (
    EntityType,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    RelationType,
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_filesystem_service_write_get_and_recall(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        record = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers and direct explanations.",
                tags=["preference"],
                provenance=provenance(),
                metadata={
                    "entities": [
                        {
                            "entity_id": "user",
                            "name": "User",
                            "entity_type": "person",
                        }
                    ]
                },
            )
        )

        fetched = await service.get(record.memory_id)
        recall = await service.passive_recall(
            PassiveRecallRequest(query="concise answers", max_results=5)
        )

        assert fetched is not None
        assert fetched.memory_id == record.memory_id
        assert len(recall.items) == 1
        assert recall.items[0].memory_id == record.memory_id

    asyncio.run(run())


def test_filesystem_service_supersede(tmp_path):
    async def run():
        service = FilesystemMemoryService(tmp_path)
        original = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers.",
                tags=["preference"],
                provenance=provenance(),
            )
        )

        replacement = await service.supersede(
            original.memory_id,
            SupersedeMemoryRequest(
                replacement=WriteMemoryRequest(
                    kind=MemoryKind.CORE_FACT,
                    title="Updated preference",
                    content="User now prefers concise answers with sparse bullets.",
                    tags=["preference"],
                    provenance=provenance(),
                )
            ),
        )

        old_record = await service.get(original.memory_id)
        assert replacement is not None
        assert old_record is not None
        assert old_record.status == RecordStatus.SUPERSEDED
        assert old_record.superseded_by == replacement.memory_id
        assert replacement.supersedes == original.memory_id
        assert (tmp_path / "memory" / "core_facts" / f"{replacement.memory_id}.md").exists()

    asyncio.run(run())


def test_async_graph_writes_do_not_block_canonical_write(tmp_path):
    class SlowLLMExtractor:
        model_name = "fake-grok"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def extract(self, record):
            self.started.set()
            await self.release.wait()
            return GraphExtractionResult(
                should_extract=True,
                rationale="grounded user preference",
                entities=[
                    ExtractedGraphEntity(
                        local_ref="user",
                        entity_type=EntityType.PERSON,
                        canonical_name="Praveen",
                        identity_candidates=[
                            GraphIdentityCandidate(
                                key_type=IdentityKeyType.NAME_VARIANT,
                                raw_value="Praveen",
                            )
                        ],
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
                        fact="Praveen loves Iron Man.",
                    )
                ],
            )

    async def run():
        llm_extractor = SlowLLMExtractor()
        service = FilesystemMemoryService(
            tmp_path,
            graph_store=InMemoryGraphStore(),
            graph_llm_extractor=llm_extractor,
            async_graph_writes=True,
        )
        await service.start_background_tasks()
        try:
            record = await service.write(
                WriteMemoryRequest(
                    kind=MemoryKind.CORE_FACT,
                    title="Marvel preference",
                    content="Praveen loves Iron Man.",
                    provenance=provenance(),
                )
            )

            stored = await service.get(record.memory_id)
            queue_counts = service.registry.graph_sync_queue_counts()

            assert stored is not None
            assert stored.memory_id == record.memory_id
            assert queue_counts["pending"] + queue_counts["running"] >= 1
            assert stored.metadata.get("graph_extraction", {}).get("mode") != "llm"

            llm_extractor.release.set()
            idle_counts = await service.wait_for_graph_queue_idle(timeout_seconds=5.0)
            hydrated = await service.get(record.memory_id)
            status = await service.health()
            graph_status = await service.get_graph_status()

            assert idle_counts["failed"] == 0
            assert hydrated is not None
            assert hydrated.metadata["graph_extraction"]["mode"] == "llm"
            assert status.graph_queue_pending_count == 0
            assert status.graph_queue_running_count == 0
            assert status.graph_queue_failed_count == 0
            assert graph_status.entity_count >= 2
            assert graph_status.relation_count >= 1
        finally:
            await service.stop_background_tasks()

    asyncio.run(run())


def test_start_background_tasks_requeues_running_graph_jobs(tmp_path):
    async def run():
        service = FilesystemMemoryService(
            tmp_path,
            graph_store=InMemoryGraphStore(),
            async_graph_writes=True,
        )
        record = MemoryRecord(
            kind=MemoryKind.CORE_FACT,
            title="Preference",
            content="Praveen likes focused engineering work.",
            provenance=provenance(),
            metadata={
                "graph_document": {
                    "memory_id": "ignored",
                    "entities": [
                        {
                            "local_ref": "user",
                            "entity_type": EntityType.PERSON.value,
                            "canonical_name": "Praveen",
                        },
                        {
                            "local_ref": "interest",
                            "entity_type": EntityType.TOPIC.value,
                            "canonical_name": "Focused engineering work",
                        },
                    ],
                    "relations": [
                        {
                            "source_ref": "user",
                            "target_ref": "interest",
                            "relation_type": RelationType.PREFERS.value,
                            "fact": "Praveen likes focused engineering work.",
                        }
                    ],
                }
            },
        )

        snapshot = await service._write_canonical_record(record, sync_passive_index=False)  # noqa: SLF001
        job = service.registry.enqueue_graph_sync(
            memory_id=record.memory_id,
            content_hash=snapshot.content_hash,
            allow_llm=False,
            persist_graph_document=True,
        )
        leased = service.registry.lease_next_graph_sync_job()

        assert leased is not None
        assert leased.job_id == job.job_id
        assert service.registry.graph_sync_queue_counts()["running"] == 1

        await service.start_background_tasks()
        try:
            idle_counts = await service.wait_for_graph_queue_idle(timeout_seconds=5.0)
            status = await service.health()

            assert idle_counts["failed"] == 0
            assert status.graph_queue_running_count == 0
            assert status.graph_relation_count >= 1
        finally:
            await service.stop_background_tasks()

    asyncio.run(run())
