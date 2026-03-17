import pytest

from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.server.app import (
    _build_embedding_service_from_env,
    _build_deterministic_graph_extractor_from_env,
    _build_entity_index_from_env,
    _build_graph_adjudicator_from_env,
    _build_graph_extractor_from_env,
    _build_ontology_curator_from_env,
    _build_graph_store_from_env,
    _build_usage_logger_from_env,
    _graph_async_writes_enabled,
    _graph_warm_cache_on_startup_enabled,
    _graph_sync_on_startup_enabled,
    _ontology_curator_interval_seconds,
    _ontology_curator_max_examples_per_group,
    _ontology_curator_max_groups,
    _ontology_curator_min_observations,
    _graph_write_retry_base_seconds,
    _graph_write_retry_max_seconds,
    _graph_write_worker_poll_seconds,
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


def test_build_usage_logger_requires_gateway_url_and_internal_token(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GATEWAY_URL", raising=False)
    monkeypatch.delenv("GATEWAY_INTERNAL_TOKEN", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_INTERNAL_TOKEN", raising=False)

    assert _build_usage_logger_from_env() is None


def test_build_usage_logger_can_construct_logger(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GATEWAY_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("GATEWAY_INTERNAL_TOKEN", "internal-token")
    monkeypatch.setenv("COSMIC_MEMORY_USAGE_TIMEOUT_SEC", "3.5")
    monkeypatch.setenv("COSMIC_MEMORY_USAGE_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("COSMIC_MEMORY_USAGE_RETRY_BASE_SECONDS", "0.25")
    monkeypatch.setenv("COSMIC_MEMORY_SERVICE_ID", "cosmic-memory:test")

    logger = _build_usage_logger_from_env()

    assert logger is not None
    assert logger.gateway_url == "http://127.0.0.1:8080"
    assert logger.internal_token == "internal-token"
    assert logger.timeout_sec == 3.5
    assert logger.max_attempts == 4
    assert logger.retry_base_seconds == 0.25
    assert logger.source_id == "cosmic-memory:test"


def test_build_graph_store_requires_neo4j_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_URI", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_NEO4J_PASSWORD", raising=False)

    with pytest.raises(RuntimeError):
        _build_graph_store_from_env(HashEmbeddingService(dimensions=32))


def test_build_graph_store_can_construct_neo4j_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    class FakeNeo4jGraphStore:
        def __init__(
            self,
            *,
            uri: str,
            username: str,
            password: str,
            database: str,
            entity_index,
            adjudicator,
            fact_adjudicator,
        ) -> None:
            captured["uri"] = uri
            captured["username"] = username
            captured["password"] = password
            captured["database"] = database
            captured["entity_index"] = entity_index
            captured["adjudicator"] = adjudicator
            captured["fact_adjudicator"] = fact_adjudicator

    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")
    monkeypatch.setenv("COSMIC_MEMORY_ENTITY_INDEX_ENABLED", "false")
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_ADJUDICATE_ENABLED", "false")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j")
    monkeypatch.setattr("cosmic_memory.server.app.Neo4jGraphStore", FakeNeo4jGraphStore)

    _build_graph_store_from_env(HashEmbeddingService(dimensions=32))

    assert captured == {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "secret",
        "database": "neo4j",
        "entity_index": None,
        "adjudicator": None,
        "fact_adjudicator": None,
    }


def test_build_entity_index_can_construct_qdrant_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeEntityIndex:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("COSMIC_MEMORY_ENTITY_INDEX_ENABLED", "true")
    monkeypatch.setenv("COSMIC_MEMORY_QDRANT_PATH", "/tmp/qdrant")
    monkeypatch.setattr("cosmic_memory.server.app.QdrantEntitySimilarityIndex", FakeEntityIndex)

    _build_entity_index_from_env(HashEmbeddingService(dimensions=32))

    assert captured["collection_name"] == "memory_entities"
    assert str(captured["path"]).endswith("qdrant_entity_data")
    assert captured["vector_size"] == 32


def test_build_entity_index_can_use_explicit_local_path(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeEntityIndex:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("COSMIC_MEMORY_ENTITY_INDEX_ENABLED", "true")
    monkeypatch.setenv("COSMIC_MEMORY_QDRANT_PATH", "/tmp/qdrant")
    monkeypatch.setenv("COSMIC_MEMORY_ENTITY_QDRANT_PATH", "/tmp/qdrant-entities")
    monkeypatch.setattr("cosmic_memory.server.app.QdrantEntitySimilarityIndex", FakeEntityIndex)

    _build_entity_index_from_env(HashEmbeddingService(dimensions=32))

    assert captured["path"] == "/tmp/qdrant-entities"


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
    monkeypatch.setenv("COSMIC_MEMORY_PRIMARY_USER_DISPLAY_NAME", "Praveen")
    monkeypatch.setattr("cosmic_memory.server.app.XAIGraphExtractionService", FakeExtractor)

    _build_graph_extractor_from_env()

    assert captured["api_key"] == "secret"
    assert captured["model_name"] == "grok-4-1-fast-reasoning"
    assert captured["timezone_name"] == "America/Chicago"
    assert captured["primary_user_display_name"] == "Praveen"


def test_build_ontology_curator_requires_xai_api_key_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_ENABLED", "true")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _build_ontology_curator_from_env()


def test_build_ontology_curator_can_construct_xai_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeCurator:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", "secret")
    monkeypatch.setenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_MODEL", "grok-4-1-fast-reasoning")
    monkeypatch.setattr("cosmic_memory.server.app.XAIOntologyCuratorService", FakeCurator)

    _build_ontology_curator_from_env()

    assert captured["api_key"] == "secret"
    assert captured["model_name"] == "grok-4-1-fast-reasoning"


def test_build_deterministic_graph_extractor_enabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_DETERMINISTIC_ENABLED", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_PRIMARY_USER_DISPLAY_NAME", "Praveen")

    extractor = _build_deterministic_graph_extractor_from_env()

    assert extractor is not None
    assert extractor.primary_user_display_name == "Praveen"


def test_build_deterministic_graph_extractor_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_DETERMINISTIC_ENABLED", "false")

    extractor = _build_deterministic_graph_extractor_from_env()

    assert extractor is None


def test_graph_sync_on_startup_defaults_to_memory_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_SYNC_ON_STARTUP", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "memory")

    assert _graph_sync_on_startup_enabled() is True


def test_graph_sync_on_startup_defaults_off_for_non_memory_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_SYNC_ON_STARTUP", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")

    assert _graph_sync_on_startup_enabled() is False


def test_graph_warm_cache_on_startup_defaults_on_for_neo4j(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_WARM_CACHE_ON_STARTUP", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")

    assert _graph_warm_cache_on_startup_enabled() is True


def test_graph_warm_cache_on_startup_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_WARM_CACHE_ON_STARTUP", "false")

    assert _graph_warm_cache_on_startup_enabled() is False


def test_graph_async_writes_default_on_for_graph_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_ASYNC_GRAPH_WRITES", raising=False)
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")

    assert _graph_async_writes_enabled() is True


def test_graph_async_writes_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_ASYNC_GRAPH_WRITES", "false")
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_BACKEND", "neo4j")

    assert _graph_async_writes_enabled() is False


def test_graph_write_tuning_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_WRITE_POLL_SECONDS", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_WRITE_RETRY_BASE_SECONDS", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_GRAPH_WRITE_RETRY_MAX_SECONDS", raising=False)

    assert _graph_write_worker_poll_seconds() == pytest.approx(0.5)
    assert _graph_write_retry_base_seconds() == pytest.approx(5.0)
    assert _graph_write_retry_max_seconds() == pytest.approx(300.0)


def test_ontology_curator_tuning_env_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_MIN_OBSERVATIONS", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_MAX_GROUPS", raising=False)
    monkeypatch.delenv("COSMIC_MEMORY_ONTOLOGY_CURATOR_MAX_EXAMPLES_PER_GROUP", raising=False)

    assert _ontology_curator_interval_seconds() is None
    assert _ontology_curator_min_observations() == 3
    assert _ontology_curator_max_groups() == 8
    assert _ontology_curator_max_examples_per_group() == 4


def test_build_graph_adjudicator_requires_xai_api_key_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_ADJUDICATE_ENABLED", "true")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _build_graph_adjudicator_from_env()


def test_build_graph_adjudicator_can_construct_xai_backend(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeAdjudicator:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_ADJUDICATE_ENABLED", "true")
    monkeypatch.setenv("XAI_API_KEY", "secret")
    monkeypatch.setenv("COSMIC_MEMORY_GRAPH_ADJUDICATE_MODEL", "grok-4-1-fast-reasoning")
    monkeypatch.setenv("COSMIC_MEMORY_TIMEZONE", "America/Chicago")
    monkeypatch.setattr("cosmic_memory.server.app.XAIEntityAdjudicationService", FakeAdjudicator)

    _build_graph_adjudicator_from_env()

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
