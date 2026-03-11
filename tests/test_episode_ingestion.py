import asyncio

from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    EpisodeObservation,
    IngestEpisodeRequest,
    MemoryProvenance,
)
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
)
from cosmic_memory.graph import (
    EntityType,
    GraphIdentityCandidate,
    IdentityKeyType,
    InMemoryGraphStore,
    RelationType,
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test", session_id="sess_20260311")


def test_ingest_episode_creates_canonical_transcript_record():
    async def run():
        service = InMemoryDevelopmentMemoryService()
        result = await service.ingest_episode(
            IngestEpisodeRequest(
                observations=[
                    EpisodeObservation(role="user", content="We should improve graph retrieval."),
                    EpisodeObservation(role="assistant", content="I will add retrieval recipes."),
                ],
                provenance=provenance(),
                tags=["session", "memory"],
                metadata={"channel": "cli"},
                extract_graph=False,
            )
        )

        assert result.observation_count == 2
        assert result.record.kind is MemoryKind.TRANSCRIPT
        assert result.record.metadata["episode_observation_count"] == 2
        assert result.record.metadata["episode_type"] == "observation"
        assert result.record.metadata["channel"] == "cli"
        assert "[user]" in result.record.content
        assert "I will add retrieval recipes." in result.record.content
        assert result.graph_episode_id is None

    asyncio.run(run())


def test_ingest_episode_can_force_graph_extraction_for_transcript():
    class FakeExtractor:
        model_name = "fake-xai"

        async def extract(self, record):
            return GraphExtractionResult(
                should_extract=True,
                rationale="episode mentions a grounded project assignment",
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
        result = await service.ingest_episode(
            IngestEpisodeRequest(
                observations=[
                    EpisodeObservation(
                        role="assistant",
                        content="Nitin Agarwal works on Cosmic Memory.",
                    )
                ],
                provenance=provenance(),
                extract_graph=True,
            )
        )

        assert result.graph_episode_id is not None
        assert "graph_document" in result.record.metadata

        active = await service.active_recall(
            ActiveRecallRequest(query="What project does nxagarwal@ualr.edu work on?")
        )

        assert active.relations
        assert active.relations[0].relation_type == RelationType.WORKS_ON.value

    asyncio.run(run())
