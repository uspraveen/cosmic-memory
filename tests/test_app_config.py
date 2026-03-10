import pytest

from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.server.app import _build_embedding_service_from_env, _build_graph_store_from_env


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
