from __future__ import annotations

import asyncio
import base64
from array import array
from types import SimpleNamespace

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import GenerateEmbeddingsRequest, MemoryProvenance, MemoryRecord
from cosmic_memory.extraction.models import ExtractedGraphEntity, GraphExtractionResult
from cosmic_memory.extraction.xai import XAIGraphExtractionService
from cosmic_memory.graph.ontology import EntityType
from cosmic_memory.ontology_curator import (
    OntologyCurationDecision,
    OntologyObservationGroup,
    XAIOntologyCuratorService,
)
from cosmic_memory.embeddings.perplexity import PerplexityStandardEmbeddingService
from cosmic_memory.usage import begin_metered_call, build_usage_event, post_usage_event


class RecordingUsageLogger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def emit(self, **kwargs) -> bool:
        self.events.append(kwargs)
        return True


def test_perplexity_embedding_service_emits_usage_per_provider_batch() -> None:
    class FakeEmbeddingsResource:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        index=index,
                        embedding=_encoded_int8_vector(dimensions=kwargs["dimensions"]),
                    )
                    for index, _text in enumerate(kwargs["input"])
                ],
                usage=SimpleNamespace(prompt_tokens=len(kwargs["input"]), total_tokens=len(kwargs["input"])),
                request_id=f"pplx_embed_req_{len(self.calls)}",
            )

    class FakeClient:
        def __init__(self) -> None:
            self.embeddings = FakeEmbeddingsResource()

        async def close(self) -> None:
            return None

    async def run() -> None:
        usage_logger = RecordingUsageLogger()
        service = PerplexityStandardEmbeddingService(
            client=FakeClient(),
            dimensions=128,
            batch_size=2,
            max_parallel_requests=1,
            usage_logger=usage_logger,  # type: ignore[arg-type]
        )
        try:
            await service.generate(
                GenerateEmbeddingsRequest(
                    texts=["alpha", "beta", "gamma"],
                    dimensions=128,
                    batch_size=2,
                    max_parallel_requests=1,
                    usage_operation="gateway.capability_wishlist.embed_item",
                    usage_source_component="gateway",
                    usage_source_id="gateway:capability_wishlist",
                    usage_request_id="req_wishlist_1",
                    usage_session_id="sess_wishlist_1",
                    usage_task_id="tsk_wishlist_1",
                    usage_route="internal",
                    usage_metadata={"wishlist_operation": "capture"},
                )
            )
        finally:
            await service.close()

        assert len(usage_logger.events) == 2
        assert usage_logger.events[0]["provider"] == "perplexity"
        assert usage_logger.events[0]["model"] == "pplx-embed-v1-4b"
        assert usage_logger.events[0]["operation"] == "gateway.capability_wishlist.embed_item"
        assert usage_logger.events[0]["source_component"] == "gateway"
        assert usage_logger.events[0]["source_id"] == "gateway:capability_wishlist"
        assert usage_logger.events[0]["request_id"] == "req_wishlist_1"
        assert usage_logger.events[0]["session_id"] == "sess_wishlist_1"
        assert usage_logger.events[0]["task_id"] == "tsk_wishlist_1"
        assert usage_logger.events[0]["route"] == "internal"
        assert usage_logger.events[0]["provider_request_id"] == "pplx_embed_req_1"
        assert getattr(usage_logger.events[0]["raw_usage"], "prompt_tokens") == 2
        assert usage_logger.events[0]["metadata_json"]["text_count"] == 2
        assert usage_logger.events[0]["metadata_json"]["wishlist_operation"] == "capture"
        assert usage_logger.events[1]["metadata_json"]["text_count"] == 1

    asyncio.run(run())


def test_xai_graph_extraction_emits_usage_with_memory_context() -> None:
    captured: dict[str, object] = {}

    class FakeChat:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def append(self, message) -> None:
            self.messages.append(message)

        def parse(self, _schema):
            return (
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_text_tokens=18, output_tokens=5, total_tokens=23),
                    request_id="xai_graph_req_1",
                ),
                GraphExtractionResult(
                    should_extract=True,
                    rationale="grounded",
                    entities=[
                        ExtractedGraphEntity(
                            local_ref="user",
                            entity_type=EntityType.PERSON,
                            canonical_name="Praveen",
                        )
                    ],
                ),
            )

    class FakeChatAPI:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeChat()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChatAPI()

        def close(self):
            return None

    async def run() -> None:
        usage_logger = RecordingUsageLogger()
        extractor = XAIGraphExtractionService(
            client=FakeClient(),
            timezone_name="America/Chicago",
            primary_user_display_name="Praveen",
            usage_logger=usage_logger,  # type: ignore[arg-type]
        )
        result = await extractor.extract(
            MemoryRecord(
                memory_id="mem_graph_1",
                kind=MemoryKind.USER_DATA,
                title="Education",
                content="I graduated from UPenn in 2022.",
                metadata={"request_id": "req_1"},
                provenance=MemoryProvenance(
                    source_kind="gateway",
                    session_id="sess_1",
                    task_id="tsk_1",
                ),
            )
        )

        assert result is not None
        assert captured["model"] == "grok-4-1-fast-reasoning"
        assert len(usage_logger.events) == 1
        event = usage_logger.events[0]
        assert event["provider"] == "xai"
        assert event["operation"] == "memory.graph_extract"
        assert event["task_id"] == "tsk_1"
        assert event["session_id"] == "sess_1"
        assert event["request_id"] == "req_1"
        assert event["provider_request_id"] == "xai_graph_req_1"
        assert getattr(event["raw_usage"], "total_tokens") == 23
        assert event["metadata_json"]["memory_id"] == "mem_graph_1"

    asyncio.run(run())


def _encoded_int8_vector(*, dimensions: int) -> str:
    values = array("b", [0 for _ in range(dimensions)])
    return base64.b64encode(values.tobytes()).decode("ascii")


def test_xai_ontology_curator_emits_usage() -> None:
    class FakeChat:
        def append(self, _message) -> None:
            return

        def parse(self, _schema):
            return (
                SimpleNamespace(
                    usage=SimpleNamespace(prompt_text_tokens=11, output_tokens=3, total_tokens=14),
                    request_id="xai_curator_req_1",
                ),
                OntologyCurationDecision(
                    decision="map_to_existing",
                    mapped_type="graduated_from",
                    confidence=0.92,
                    rationale="consistent recurring education phrasing",
                ),
            )

    class FakeChatAPI:
        def create(self, **kwargs):
            del kwargs
            return FakeChat()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChatAPI()

        def close(self):
            return None

    async def run() -> None:
        usage_logger = RecordingUsageLogger()
        curator = XAIOntologyCuratorService(
            client=FakeClient(),
            usage_logger=usage_logger,  # type: ignore[arg-type]
        )
        decision = await curator.curate_group(
            OntologyObservationGroup(
                observation_kind="relation_type",
                alias_label="alma_mater",
                observation_count=4,
                fit_counts={"weak": 4},
                fallback_type_counts={"part_of": 4},
                example_evidence=["graduated from upenn in 2022"],
            )
        )
        assert decision.decision == "map_to_existing"
        assert len(usage_logger.events) == 1
        event = usage_logger.events[0]
        assert event["operation"] == "memory.ontology_curate"
        assert event["provider_request_id"] == "xai_curator_req_1"
        assert getattr(event["raw_usage"], "total_tokens") == 14
        assert event["metadata_json"]["alias_label"] == "alma_mater"

    asyncio.run(run())


def test_memory_usage_post_accepts_202_response() -> None:
    class FakeResponse:
        status_code = 202

        def raise_for_status(self) -> None:
            return

    class FakeClient:
        async def post(self, *args, **kwargs):
            del args, kwargs
            return FakeResponse()

    async def run() -> None:
        event = build_usage_event(
            metered_call=begin_metered_call(prefix="call"),
            source_component="session_manager",
            source_id="cosmic-memory",
            provider="xai",
            model="grok-4-1-fast-reasoning",
            usage_kind="chat_completion",
            operation="memory.graph_extract",
            raw_usage=SimpleNamespace(prompt_text_tokens=12, output_tokens=4, total_tokens=16),
        )
        posted = await post_usage_event(
            client=FakeClient(),
            gateway_url="http://127.0.0.1:8080",
            internal_token="internal-token",
            event=event,
        )
        assert posted is True

    asyncio.run(run())
