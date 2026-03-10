import asyncio
import os

import pytest

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    MemoryProvenance,
    PassiveRecallRequest,
    WriteMemoryRequest,
)
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import Neo4jGraphStore


pytestmark = pytest.mark.skipif(
    os.environ.get("COSMIC_MEMORY_RUN_NEO4J_INTEGRATION") != "1",
    reason="Set COSMIC_MEMORY_RUN_NEO4J_INTEGRATION=1 with Neo4j env vars to run this test.",
)


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="integration_test", created_by="pytest")


def test_neo4j_graph_store_supports_passive_and_active_recall(tmp_path):
    pytest.importorskip("neo4j")

    async def run():
        graph_store = Neo4jGraphStore(
            uri=os.environ["COSMIC_MEMORY_NEO4J_URI"],
            username=os.environ["COSMIC_MEMORY_NEO4J_USERNAME"],
            password=os.environ["COSMIC_MEMORY_NEO4J_PASSWORD"],
            database=os.environ.get("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j"),
        )
        await _wipe_graph(graph_store)
        try:
            service = FilesystemMemoryService(tmp_path, graph_store=graph_store)
            record = await service.write(
                WriteMemoryRequest(
                    kind=MemoryKind.AGENT_NOTE,
                    title="Project relationship",
                    content="Nitin Agarwal works on Cosmic Memory.",
                    provenance=provenance(),
                    metadata={
                        "graph_document": {
                            "memory_id": "ignored",
                            "entities": [
                                {
                                    "local_ref": "person",
                                    "entity_type": "person",
                                    "canonical_name": "Nitin Agarwal",
                                    "identity_candidates": [
                                        {
                                            "key_type": "email",
                                            "raw_value": "nxagarwal@ualr.edu",
                                        }
                                    ],
                                },
                                {
                                    "local_ref": "project",
                                    "entity_type": "project",
                                    "canonical_name": "Cosmic Memory",
                                },
                            ],
                            "relations": [
                                {
                                    "source_ref": "person",
                                    "target_ref": "project",
                                    "relation_type": "works_on",
                                    "fact": "Nitin Agarwal works on Cosmic Memory.",
                                }
                            ],
                        }
                    },
                )
            )

            passive = await service.passive_recall(
                PassiveRecallRequest(
                    query="What project is nxagarwal@ualr.edu working on?",
                    max_results=4,
                    token_budget=120,
                    include_diagnostics=True,
                )
            )
            active = await service.active_recall(
                ActiveRecallRequest(
                    query="What project does nxagarwal@ualr.edu work on?",
                    include_diagnostics=True,
                )
            )

            assert passive.items
            assert passive.items[0].memory_id == record.memory_id
            assert passive.diagnostics is not None
            assert passive.diagnostics.flags["graph_assist_used"] is True

            assert active.items
            assert active.items[0].memory_id == record.memory_id
            assert active.relations
            assert active.relations[0].relation_type == "works_on"
            assert active.diagnostics is not None
            assert active.diagnostics.flags["graph_used"] is True
        finally:
            await _wipe_graph(graph_store)
            await graph_store.close()

    asyncio.run(run())


async def _wipe_graph(graph_store: Neo4jGraphStore) -> None:
    async with graph_store.driver.session(database=graph_store.database) as session:
        result = await session.run("MATCH (n) DETACH DELETE n")
        await result.consume()
