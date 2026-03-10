import asyncio

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import MemoryProvenance, PassiveRecallRequest, WriteMemoryRequest
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
