import pytest

from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.server.app import (
    _build_embedding_service_from_env,
    _build_graph_extractor_from_env,
    _build_graph_store_from_env,
    _build_passive_index_from_env,
)


def test_build_embedding_service_requires_api_key_in_production(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _build_embedding_service_from_env(require_remote=True)


def test_build_embedding_service_can_use_dev_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("PPLX_API_KEY", raising=False)

    service = _build_embedding_service_from_env(require_remote=False)

    assert isinstance(service, HashEmbeddingService)


def test_build_graph_store_requires_neo4j_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_URI", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_PASSWORD", raising=False)

    with pytest.raises(RuntimeError):
        _build_graph_store_from_env()


def test_build_graph_store_can_construct_neo4j_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    class FakeNeo4jGraphStore:
        def __init__(self, *, uri: str, username: str, password: str, database: str) -> None:
            captured["uri"] = uri
            captured["username"] = username
            captured["password"] = password
            captured["database"] = database

    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j")
    monkeypatch.setattr("cosmic_memory.server.app.Neo4jGraphStore", FakeNeo4jGraphStore)

    _build_graph_store_from_env()

    assert captured == {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "secret",
        "database": "neo4j",
    }


def test_build_graph_extractor_requires_xai_api_key_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_EXTRACT_ENABLED", "true")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _build_graph_extractor_from_env()


def test_build_graph_extractor_can_construct_xai_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeExtractor:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_EXTRACT_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", "secret")
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_EXTRACT_MODEL", "grok-4-1-fast-reasoning")
    monkeypatch.setenv("COSMIC_MEMORY_TIMEZONE", "America/Chicago")
    monkeypatch.setattr("cosmic_memory.server.app.XAIGraphExtractionService", FakeExtractor)

    _build_graph_extractor_from_env()

    assert captured["api_key"] == "secret"
    assert captured["model_name"] == "grok-4-1-fast-reasoning"
    assert captured["timezone_name"] == "America/Chicago"


def test_build_passive_index_prefers_fastembed_for_local_qdrant(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeFastEmbedSparseEncoder:
        def __init__(self, model_name: str = "Qdrant/bm25") -> None:
            self.model_name = model_name

    class FakeIndex:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.delenv("COSMIC_MEMORY_QDRANT_URL", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_QDRANT_PATH", "/tmp/qdrant")
    monkeypatch.delenv("COSMIC_MEMORY_SPARSE_BACKEND", raising=False)
    monkeypatch.setattr("cosmic_memory.server.app.FastEmbedSparseEncoder", FakeFastEmbedSparseEncoder)
    monkeypatch.setattr("cosmic_memory.server.app.QdrantHybridMemoryIndex", FakeIndex)

    _build_passive_index_from_env(HashEmbeddingService(dimensions=32))

    assert isinstance(captured["sparse_encoder"], FakeFastEmbedSparseEncoder)


def test_build_passive_index_can_force_native_sparse_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeIndex:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.delenv("COSMIC_MEMORY_QDRANT_URL", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_QDRANT_PATH", "/tmp/qdrant")
    monkeypatch.setenv("COSMIC_MEMORY_SPARSE_BACKEND", "native")
    monkeypatch.setattr("cosmic_memory.server.app.QdrantHybridMemoryIndex", FakeIndex)

    _build_passive_index_from_env(HashEmbeddingService(dimensions=32))

    assert captured["sparse_encoder"] is None
