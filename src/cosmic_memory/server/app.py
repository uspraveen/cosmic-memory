"""FastAPI server for cosmic-memory."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status

from cosmic_memory.control_surface import (
    CurrentStateRequest,
    MemoryBriefRequest,
    MemoryQueryPlanRequest,
    ResolveIdentityRequest,
    TemporalFactsRequest,
)
from cosmic_memory.dev_service import InMemoryDevelopmentMemoryService
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    GenerateEmbeddingsRequest,
    GraphSyncRequest,
    IngestEpisodeRequest,
    PassiveRecallRequest,
    SupersedeMemoryRequest,
    WriteCoreFactRequest,
    WriteMemoryRequest,
)
from cosmic_memory.embeddings import HashEmbeddingService, PerplexityStandardEmbeddingService
from cosmic_memory.embeddings.base import EmbeddingService
from cosmic_memory.env import load_env_file
from cosmic_memory.extraction import DeterministicGraphExtractionService, XAIGraphExtractionService
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import (
    InMemoryGraphStore,
    Neo4jGraphStore,
    XAIEntityAdjudicationService,
    XAIFactAdjudicationService,
)
from cosmic_memory.graph.entity_index import InMemoryEntitySimilarityIndex
from cosmic_memory.graph.entity_qdrant import QdrantEntitySimilarityIndex
from cosmic_memory.index import FastEmbedSparseEncoder, QdrantHybridMemoryIndex, SimpleSparseEncoder
from cosmic_memory.service import MemoryService

logger = logging.getLogger(__name__)


def create_app(
    service: MemoryService,
    *,
    embedding_service: EmbeddingService,
) -> FastAPI:
    internal_token = (
        os.environ.get("COSMIC_MEMORY_INTERNAL_TOKEN")
        or os.environ.get("GATEWAY_INTERNAL_TOKEN")
        or ""
    ).strip()

    async def require_internal_token(
        x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
    ) -> None:
        if not internal_token:
            return
        if x_internal_token == internal_token:
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token.",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if _sync_on_startup_enabled():
            sync_index = getattr(app.state.memory_service, "sync_index", None)
            if sync_index is not None:
                await sync_index()
        if _graph_sync_on_startup_enabled():
            sync_graph = getattr(app.state.memory_service, "sync_graph", None)
            if sync_graph is not None:
                try:
                    await sync_graph()
                except Exception:
                    logger.exception("cosmic_memory.graph_sync_on_startup_failed")
        if _graph_warm_cache_on_startup_enabled():
            warm_graph_cache = getattr(app.state.memory_service, "warm_graph_cache", None)
            if warm_graph_cache is not None:
                try:
                    await warm_graph_cache()
                except Exception:
                    logger.exception("cosmic_memory.graph_cache_warm_on_startup_failed")
        yield
        await _close_if_present(getattr(app.state, "embedding_service", None))
        passive_index = getattr(getattr(app.state, "memory_service", None), "passive_index", None)
        await _close_if_present(passive_index)
        graph_extractor = getattr(getattr(app.state, "memory_service", None), "graph_extractor", None)
        await _close_if_present(graph_extractor)
        graph_store = getattr(getattr(app.state, "memory_service", None), "graph_store", None)
        await _close_if_present(graph_store)

    app = FastAPI(title="cosmic-memory", version="0.1.0", lifespan=lifespan)
    app.state.memory_service = service
    app.state.embedding_service = embedding_service

    @app.get("/health")
    async def health(request: Request):
        svc: MemoryService = request.app.state.memory_service
        return await svc.health()

    @app.post("/v1/embeddings/generate")
    async def generate_embeddings(
        payload: GenerateEmbeddingsRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        embedder: EmbeddingService = request.app.state.embedding_service
        return await embedder.generate(payload)

    @app.post("/v1/memories", status_code=status.HTTP_201_CREATED)
    async def write_memory(
        payload: WriteMemoryRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.write(payload)

    @app.post("/v1/episodes", status_code=status.HTTP_201_CREATED)
    async def ingest_episode(
        payload: IngestEpisodeRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.ingest_episode(payload)

    @app.post("/v1/core-facts", status_code=status.HTTP_201_CREATED)
    async def write_core_fact(
        payload: WriteCoreFactRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.write_core_fact(payload)

    @app.get("/v1/core-facts")
    async def get_core_fact_block(
        request: Request,
        limit: int | None = None,
        max_chars: int = 1500,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.build_core_fact_block(limit=limit, max_chars=max_chars)

    @app.get("/v1/memories/{memory_id}")
    async def get_memory(
        memory_id: str,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        record = await svc.get(memory_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
        return record

    @app.post("/v1/query/passive")
    async def passive_recall(
        payload: PassiveRecallRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.passive_recall(payload)

    @app.post("/v1/query/active")
    async def active_recall(
        payload: ActiveRecallRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.active_recall(payload)

    @app.get("/v1/agent/schema-context")
    async def get_schema_context(
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.get_schema_context()

    @app.post("/v1/agent/plan")
    async def plan_query(
        payload: MemoryQueryPlanRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.plan_query(payload)

    @app.post("/v1/agent/resolve-identity")
    async def resolve_identity(
        payload: ResolveIdentityRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.resolve_identity(payload)

    @app.post("/v1/agent/current-state")
    async def current_state(
        payload: CurrentStateRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.get_current_state(payload)

    @app.post("/v1/agent/temporal-facts")
    async def temporal_facts(
        payload: TemporalFactsRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.get_temporal_facts(payload)

    @app.post("/v1/agent/memory-brief")
    async def memory_brief(
        payload: MemoryBriefRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.build_memory_brief(payload)

    @app.get("/v1/index/status")
    async def get_index_status(
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.get_index_status()

    @app.post("/v1/index/sync")
    async def sync_index(
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.sync_index()

    @app.post("/v1/index/rebuild")
    async def rebuild_index(
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.rebuild_index()

    @app.get("/v1/graph/status")
    async def get_graph_status(
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.get_graph_status()

    @app.post("/v1/graph/sync")
    async def sync_graph(
        request: Request,
        payload: GraphSyncRequest | None = None,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.sync_graph(payload or GraphSyncRequest())

    @app.post("/v1/graph/rebuild")
    async def rebuild_graph(
        request: Request,
        payload: GraphSyncRequest | None = None,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        return await svc.rebuild_graph(payload or GraphSyncRequest())

    @app.post("/v1/memories/{memory_id}/supersede")
    async def supersede_memory(
        memory_id: str,
        payload: SupersedeMemoryRequest,
        request: Request,
        _: None = Depends(require_internal_token),
    ):
        svc: MemoryService = request.app.state.memory_service
        record = await svc.supersede(memory_id, payload)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
        return record

    return app


def create_default_app() -> FastAPI:
    return create_filesystem_app()


def create_development_app() -> FastAPI:
    embedding_service = HashEmbeddingService()
    return create_app(
        InMemoryDevelopmentMemoryService(
            graph_store=InMemoryGraphStore(
                entity_index=InMemoryEntitySimilarityIndex(embedding_service=embedding_service)
            ),
            graph_extractor=DeterministicGraphExtractionService(),
        ),
        embedding_service=embedding_service,
    )


def create_filesystem_app(root_dir: str | None = None) -> FastAPI:
    load_env_file()
    data_root = root_dir or os.environ.get("COSMIC_MEMORY_DATA_DIR", ".cosmic-memory-data")
    embedding_service = _build_embedding_service_from_env(require_remote=True)
    passive_index = _build_passive_index_from_env(
        embedding_service,
        default_path=str(Path(data_root) / "qdrant_data"),
    )
    graph_store = _build_graph_store_from_env(
        embedding_service,
        default_path=str(Path(data_root) / "qdrant_data"),
    )
    graph_extractor = _build_deterministic_graph_extractor_from_env()
    graph_llm_extractor = _build_graph_extractor_from_env()
    return create_app(
        FilesystemMemoryService(
            data_root,
            passive_index=passive_index,
            graph_store=graph_store,
            graph_extractor=graph_extractor,
            graph_llm_extractor=graph_llm_extractor,
            passive_graph_timeout_seconds=float(
                os.environ.get("COSMIC_MEMORY_PASSIVE_GRAPH_TIMEOUT_MS", "120")
            )
            / 1000.0,
        ),
        embedding_service=embedding_service,
    )


async def _close_if_present(resource) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    await close()


def _build_embedding_service_from_env(*, require_remote: bool = True) -> EmbeddingService:
    dimensions = int(os.environ.get("COSMIC_MEMORY_EMBEDDING_DIMENSIONS", "1024"))
    model_name = os.environ.get("COSMIC_MEMORY_EMBEDDING_MODEL", "pplx-embed-v1-4b")
    batch_size = int(os.environ.get("COSMIC_MEMORY_EMBED_BATCH_SIZE", "128"))
    max_parallel_requests = int(os.environ.get("COSMIC_MEMORY_EMBED_MAX_PARALLEL", "4"))
    encoding_format = os.environ.get("COSMIC_MEMORY_EMBED_ENCODING", "base64_int8")
    normalize = os.environ.get("COSMIC_MEMORY_EMBED_NORMALIZE", "true").lower() != "false"
    max_retries = int(os.environ.get("COSMIC_MEMORY_EMBED_MAX_RETRIES", "4"))
    retry_base_seconds = float(os.environ.get("COSMIC_MEMORY_EMBED_RETRY_BASE_SECONDS", "0.75"))
    retry_max_seconds = float(os.environ.get("COSMIC_MEMORY_EMBED_RETRY_MAX_SECONDS", "8.0"))
    api_key = os.environ.get("PERPLEXITY_API_KEY") or os.environ.get("PPLX_API_KEY")
    if not api_key:
        if not require_remote:
            return HashEmbeddingService(dimensions=dimensions)
        raise RuntimeError(
            "PERPLEXITY_API_KEY is required for production app factories. "
            "Use create_development_app() for local deterministic testing."
        )
    return PerplexityStandardEmbeddingService(
        api_key=api_key,
        model_name=model_name,
        dimensions=dimensions,
        batch_size=batch_size,
        max_parallel_requests=max_parallel_requests,
        encoding_format=encoding_format,
        normalize=normalize,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        retry_max_seconds=retry_max_seconds,
    )


def _build_passive_index_from_env(
    embedding_service: EmbeddingService,
    *,
    default_path: str | None = None,
):
    qdrant_url = os.environ.get("COSMIC_MEMORY_QDRANT_URL")
    qdrant_path = (
        os.environ.get("COSMIC_MEMORY_QDRANT_PATH")
        or os.environ.get("QDRANT_PATH")
        or default_path
    )
    sparse_backend = os.environ.get("COSMIC_MEMORY_SPARSE_BACKEND", "auto").strip().lower()
    sparse_model_name = os.environ.get("COSMIC_MEMORY_SPARSE_MODEL", "Qdrant/bm25")

    sparse_encoder = None
    if sparse_backend == "simple":
        sparse_encoder = SimpleSparseEncoder()
    elif sparse_backend == "fastembed":
        sparse_encoder = FastEmbedSparseEncoder(model_name=sparse_model_name)
    elif sparse_backend in {"", "auto"} and qdrant_path and not qdrant_url:
        sparse_encoder = FastEmbedSparseEncoder(model_name=sparse_model_name)
    elif sparse_backend not in {"", "auto", "native"}:
        raise RuntimeError(
            "Unsupported COSMIC_MEMORY_SPARSE_BACKEND value. "
            "Supported values are `auto`, `native`, `fastembed`, and `simple`."
        )

    return QdrantHybridMemoryIndex(
        embedding_service=embedding_service,
        sparse_encoder=sparse_encoder,
        sparse_model_name=sparse_model_name,
        collection_name=os.environ.get("COSMIC_MEMORY_QDRANT_COLLECTION", "memories"),
        vector_size=int(os.environ.get("COSMIC_MEMORY_VECTOR_SIZE", str(embedding_service.dimensions))),
        url=qdrant_url,
        path=qdrant_path,
        embed_batch_size=int(os.environ.get("COSMIC_MEMORY_EMBED_BATCH_SIZE", "128")),
        embed_parallel_requests=int(os.environ.get("COSMIC_MEMORY_EMBED_MAX_PARALLEL", "4")),
        dense_encoding_format=os.environ.get("COSMIC_MEMORY_EMBED_ENCODING", "base64_int8"),
    )


def _sync_on_startup_enabled() -> bool:
    raw = os.environ.get("COSMIC_MEMORY_SYNC_ON_STARTUP") or os.environ.get(
        "MEMORY_SYNC_ON_STARTUP",
        "true",
    )
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _graph_sync_on_startup_enabled() -> bool:
    raw = os.environ.get("COSMIC_MEMORY_GRAPH_SYNC_ON_STARTUP")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    backend = os.environ.get("COSMIC_MEMORY_GRAPH_BACKEND", "none").strip().lower()
    return backend == "memory"


def _graph_warm_cache_on_startup_enabled() -> bool:
    raw = os.environ.get("COSMIC_MEMORY_GRAPH_WARM_CACHE_ON_STARTUP")
    if raw is not None:
        return raw.strip().lower() not in {"0", "false", "no", "off"}
    backend = os.environ.get("COSMIC_MEMORY_GRAPH_BACKEND", "none").strip().lower()
    return backend == "neo4j"


def _build_graph_store_from_env(
    embedding_service: EmbeddingService,
    *,
    default_path: str | None = None,
):
    backend = os.environ.get("COSMIC_MEMORY_GRAPH_BACKEND", "none").strip().lower()
    if backend in {"", "none", "off"}:
        return None
    adjudicator = _build_graph_adjudicator_from_env()
    fact_adjudicator = _build_graph_fact_adjudicator_from_env()
    entity_index = _build_entity_index_from_env(
        embedding_service,
        default_path=default_path,
    )
    if backend == "memory":
        return InMemoryGraphStore(
            entity_index=entity_index,
            adjudicator=adjudicator,
            fact_adjudicator=fact_adjudicator,
        )
    if backend == "neo4j":
        uri = os.environ.get("COSMIC_MEMORY_NEO4J_URI")
        username = os.environ.get("COSMIC_MEMORY_NEO4J_USERNAME")
        password = os.environ.get("COSMIC_MEMORY_NEO4J_PASSWORD")
        database = os.environ.get("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j")
        if not uri or not username or not password:
            raise RuntimeError(
                "COSMIC_MEMORY_NEO4J_URI, COSMIC_MEMORY_NEO4J_USERNAME, and "
                "COSMIC_MEMORY_NEO4J_PASSWORD are required when "
                "COSMIC_MEMORY_GRAPH_BACKEND=neo4j."
            )
        return Neo4jGraphStore(
            uri=uri,
            username=username,
            password=password,
            database=database,
            entity_index=entity_index,
            adjudicator=adjudicator,
            fact_adjudicator=fact_adjudicator,
        )
    raise RuntimeError(
        f"Unsupported graph backend: {backend}. "
        "Supported values are `none`, `memory`, and `neo4j`."
    )


def _build_entity_index_from_env(
    embedding_service: EmbeddingService,
    *,
    default_path: str | None = None,
):
    enabled = os.environ.get("COSMIC_MEMORY_ENTITY_INDEX_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    qdrant_url = os.environ.get("COSMIC_MEMORY_QDRANT_URL")
    explicit_entity_path = os.environ.get("COSMIC_MEMORY_ENTITY_QDRANT_PATH")
    shared_path = (
        os.environ.get("COSMIC_MEMORY_QDRANT_PATH")
        or os.environ.get("QDRANT_PATH")
        or default_path
    )
    if explicit_entity_path:
        qdrant_path = explicit_entity_path
    elif qdrant_url:
        qdrant_path = shared_path
    elif shared_path:
        qdrant_path = str(Path(shared_path).with_name("qdrant_entity_data"))
    else:
        qdrant_path = None
    return QdrantEntitySimilarityIndex(
        embedding_service=embedding_service,
        collection_name=os.environ.get("COSMIC_MEMORY_ENTITY_COLLECTION", "memory_entities"),
        vector_size=int(os.environ.get("COSMIC_MEMORY_VECTOR_SIZE", str(embedding_service.dimensions))),
        url=qdrant_url,
        path=qdrant_path,
        embed_batch_size=int(os.environ.get("COSMIC_MEMORY_EMBED_BATCH_SIZE", "128")),
        embed_parallel_requests=int(os.environ.get("COSMIC_MEMORY_EMBED_MAX_PARALLEL", "4")),
        dense_encoding_format=os.environ.get("COSMIC_MEMORY_EMBED_ENCODING", "base64_int8"),
    )


def _build_graph_extractor_from_env():
    raw_enabled = os.environ.get("COSMIC_MEMORY_GRAPH_EXTRACT_ENABLED")
    api_key = os.environ.get("XAI_API_KEY")
    if raw_enabled is None:
        enabled = bool(api_key)
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is required when graph extraction is enabled."
        )
    return XAIGraphExtractionService(
        api_key=api_key,
        model_name=os.environ.get(
            "COSMIC_MEMORY_GRAPH_EXTRACT_MODEL",
            "grok-4-1-fast-reasoning",
        ),
        timezone_name=os.environ.get("COSMIC_MEMORY_TIMEZONE", "UTC"),
        primary_user_display_name=os.environ.get("COSMIC_MEMORY_PRIMARY_USER_DISPLAY_NAME"),
        max_parallel_requests=int(
            os.environ.get("COSMIC_MEMORY_GRAPH_EXTRACT_MAX_PARALLEL", "2")
        ),
        max_retries=int(os.environ.get("COSMIC_MEMORY_GRAPH_EXTRACT_MAX_RETRIES", "3")),
        retry_base_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_EXTRACT_RETRY_BASE_SECONDS", "1.0")
        ),
        retry_max_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_EXTRACT_RETRY_MAX_SECONDS", "12.0")
        ),
    )


def _build_deterministic_graph_extractor_from_env():
    raw_enabled = os.environ.get("COSMIC_MEMORY_GRAPH_DETERMINISTIC_ENABLED", "true")
    if raw_enabled.strip().lower() in {"0", "false", "no", "off"}:
        return None
    return DeterministicGraphExtractionService(
        primary_user_display_name=os.environ.get("COSMIC_MEMORY_PRIMARY_USER_DISPLAY_NAME")
    )


def _build_graph_adjudicator_from_env():
    raw_enabled = os.environ.get("COSMIC_MEMORY_GRAPH_ADJUDICATE_ENABLED")
    api_key = os.environ.get("XAI_API_KEY")
    if raw_enabled is None:
        enabled = bool(api_key)
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is required when graph adjudication is enabled."
        )
    return XAIEntityAdjudicationService(
        api_key=api_key,
        model_name=os.environ.get(
            "COSMIC_MEMORY_GRAPH_ADJUDICATE_MODEL",
            "grok-4-1-fast-reasoning",
        ),
        timezone_name=os.environ.get("COSMIC_MEMORY_TIMEZONE", "UTC"),
        max_parallel_requests=int(
            os.environ.get("COSMIC_MEMORY_GRAPH_ADJUDICATE_MAX_PARALLEL", "2")
        ),
        max_retries=int(
            os.environ.get("COSMIC_MEMORY_GRAPH_ADJUDICATE_MAX_RETRIES", "3")
        ),
        retry_base_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_ADJUDICATE_RETRY_BASE_SECONDS", "1.0")
        ),
        retry_max_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_ADJUDICATE_RETRY_MAX_SECONDS", "12.0")
        ),
    )


def _build_graph_fact_adjudicator_from_env():
    raw_enabled = os.environ.get("COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_ENABLED")
    api_key = os.environ.get("XAI_API_KEY")
    if raw_enabled is None:
        enabled = bool(api_key)
    else:
        enabled = raw_enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return None
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is required when graph fact adjudication is enabled."
        )
    return XAIFactAdjudicationService(
        api_key=api_key,
        model_name=os.environ.get(
            "COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_MODEL",
            "grok-4-1-fast-reasoning",
        ),
        timezone_name=os.environ.get("COSMIC_MEMORY_TIMEZONE", "UTC"),
        max_parallel_requests=int(
            os.environ.get("COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_MAX_PARALLEL", "2")
        ),
        max_retries=int(
            os.environ.get("COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_MAX_RETRIES", "3")
        ),
        retry_base_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_RETRY_BASE_SECONDS", "1.0")
        ),
        retry_max_seconds=float(
            os.environ.get("COSMIC_MEMORY_GRAPH_FACT_ADJUDICATE_RETRY_MAX_SECONDS", "12.0")
        ),
    )
