import asyncio

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    EpisodeObservation,
    IngestEpisodeRequest,
    MemoryProvenance,
    PassiveRecallRequest,
    WriteMemoryRequest,
)
from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.index.qdrant import QdrantHybridMemoryIndex
from cosmic_memory.index.sparse import SimpleSparseEncoder


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="gateway", created_by="test")


def test_qdrant_local_path_smoke(tmp_path):
    async def run():
        index = QdrantHybridMemoryIndex(
            embedding_service=HashEmbeddingService(dimensions=128),
            sparse_encoder=SimpleSparseEncoder(),
            path=str(tmp_path / "qdrant"),
            vector_size=128,
        )
        service = FilesystemMemoryService(tmp_path, passive_index=index, index_sync_batch_size=32)
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.CORE_FACT,
                title="Preference",
                content="User prefers concise answers and short summaries.",
                tags=["preference"],
                provenance=provenance(),
            )
        )
        await service.write(
            WriteMemoryRequest(
                kind=MemoryKind.USER_DATA,
                title="Large import",
                content="Imported PDF about completely unrelated gardening topics.",
                tags=["imported"],
                provenance=provenance(),
            )
        )

        result = await service.passive_recall(
            PassiveRecallRequest(query="concise short summaries", max_results=3, token_budget=50)
        )

        assert result.items
        assert result.items[0].kind == MemoryKind.CORE_FACT
        await index.close()

    asyncio.run(run())


def test_qdrant_local_path_handles_concurrent_episode_ingest(tmp_path):
    async def run():
        index = QdrantHybridMemoryIndex(
            embedding_service=HashEmbeddingService(dimensions=128),
            sparse_encoder=SimpleSparseEncoder(),
            path=str(tmp_path / "qdrant"),
            vector_size=128,
        )
        service = FilesystemMemoryService(tmp_path, passive_index=index, index_sync_batch_size=32)

        async def ingest(turn: int):
            return await service.ingest_episode(
                IngestEpisodeRequest(
                    observations=[
                        EpisodeObservation(role="user", content=f"Question {turn}?"),
                        EpisodeObservation(role="assistant", content=f"Answer {turn}."),
                    ],
                    provenance=provenance(),
                    extract_graph=False,
                )
            )

        results = await asyncio.gather(*(ingest(turn) for turn in range(6)))

        assert len({result.record.memory_id for result in results}) == 6
        recall = await service.passive_recall(
            PassiveRecallRequest(query="Answer 5", max_results=10, token_budget=200)
        )
        assert any("Answer 5." in item.content for item in recall.items)
        await index.close()

    asyncio.run(run())
