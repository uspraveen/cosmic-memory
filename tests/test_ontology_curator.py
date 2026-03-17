import asyncio

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, WriteMemoryRequest
from cosmic_memory.extraction.models import (
    ExtractedGraphEntity,
    ExtractedGraphRelation,
    GraphExtractionResult,
    OntologyObservation,
)
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import EntityType, RelationType
from cosmic_memory.ontology_curator import OntologyCurationDecision


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


class AliasAwareEducationExtractor:
    model_name = "fake-grok"

    def __init__(self) -> None:
        self._alias_provider = None

    def set_ontology_alias_provider(self, provider) -> None:
        self._alias_provider = provider

    async def extract(self, record):
        aliases = list(self._alias_provider() if self._alias_provider is not None else ())
        has_alma_mater_alias = any(
            alias.observation_kind == "relation_type"
            and alias.alias_label == "alma_mater"
            and alias.mapped_type == "graduated_from"
            for alias in aliases
        )
        observations = []
        if not has_alma_mater_alias:
            observations.append(
                OntologyObservation(
                    observation_kind="relation_type",
                    observed_label="alma_mater",
                    fallback_type="graduated_from",
                    fit_level="weak",
                    confidence=0.86,
                    rationale="Educational background phrasing is semantically closer to alma_mater.",
                    evidence=record.content,
                )
            )
        return GraphExtractionResult(
            should_extract=True,
            rationale="capture graduation fact",
            entities=[
                ExtractedGraphEntity(
                    local_ref="user",
                    entity_type=EntityType.PERSON,
                    canonical_name="Praveen",
                ),
                ExtractedGraphEntity(
                    local_ref="school",
                    entity_type=EntityType.ORGANIZATION,
                    canonical_name="University of Pennsylvania",
                    alias_values=["UPenn"],
                ),
            ],
            relations=[
                ExtractedGraphRelation(
                    source_ref="user",
                    target_ref="school",
                    relation_type=RelationType.GRADUATED_FROM,
                    fact=record.content,
                )
            ],
            ontology_observations=observations,
        )


class FakeOntologyCurator:
    model_name = "fake-curator"

    async def curate_group(self, group, *, learned_aliases=()):
        assert group.alias_label == "alma_mater"
        return OntologyCurationDecision(
            decision="map_to_existing",
            mapped_type="graduated_from",
            confidence=0.94,
            rationale="Recurring education phrasing should map to graduated_from.",
        )

    async def close(self) -> None:
        return None


def test_manual_ontology_curation_reuses_deferred_observations_and_teaches_aliases(tmp_path):
    async def run():
        extractor = AliasAwareEducationExtractor()
        service = FilesystemMemoryService(
            tmp_path,
            graph_llm_extractor=extractor,
            ontology_curator=FakeOntologyCurator(),
            ontology_curator_min_observations=2,
        )

        first = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Education one",
                content="I graduated from University of Pennsylvania (UPenn) in 2022.",
                provenance=provenance(),
            )
        )
        first_status = await service.get_ontology_status()
        first_curate = await service.curate_ontology()

        assert first is not None
        assert first_status.pending_observation_count == 1
        assert first_curate.enabled is True
        assert first_curate.alias_upserts == 0
        assert first_curate.deferred_group_count == 1

        second = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Education two",
                content="Praveen graduated from UPenn in 2022.",
                provenance=provenance(),
            )
        )
        second_curate = await service.curate_ontology()
        second_status = await service.get_ontology_status()

        assert second is not None
        assert second_curate.alias_upserts == 1
        assert second_curate.decisions[0].mapped_type == "graduated_from"
        assert second_status.pending_observation_count == 0
        assert second_status.deferred_observation_count == 0
        assert second_status.active_alias_count == 1

        third = await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Education three",
                content="I finished at UPenn in 2022.",
                provenance=provenance(),
            )
        )
        third_record = await service.get(third.memory_id)
        final_status = await service.get_ontology_status()

        assert third_record is not None
        assert third_record.metadata["graph_extraction"]["mode"] == "llm"
        assert third_record.metadata["graph_extraction"]["ontology_observations"] == []
        assert final_status.pending_observation_count == 0
        assert final_status.active_alias_count == 1

    asyncio.run(run())


def test_scheduled_ontology_curation_can_promote_aliases_in_background(tmp_path):
    async def run():
        service = FilesystemMemoryService(
            tmp_path,
            graph_llm_extractor=AliasAwareEducationExtractor(),
            ontology_curator=FakeOntologyCurator(),
            ontology_curator_interval_seconds=0.05,
            ontology_curator_min_observations=2,
        )
        await service.start_background_tasks()
        try:
            for index in range(2):
                await service.write(
                    WriteMemoryRequest(
                        kind=MemoryKind.USER_DATA,
                        title=f"Education {index}",
                        content="I graduated from University of Pennsylvania (UPenn) in 2022.",
                        provenance=provenance(),
                    )
                )

            deadline = asyncio.get_running_loop().time() + 3.0
            status = await service.get_ontology_status()
            while status.active_alias_count == 0 and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
                status = await service.get_ontology_status()

            health = await service.health()

            assert status.active_alias_count == 1
            assert status.last_run_status == "succeeded"
            assert health.ontology_curator_enabled is True
            assert health.ontology_alias_count == 1
        finally:
            await service.stop_background_tasks()

    asyncio.run(run())
