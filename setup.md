# Setup

This repo uses `pyproject.toml` as the canonical package and dependency definition.
`requirements.txt` is only a convenience entry point for environments that require it.

## Prerequisites

- Python `3.11+`
- `pip`
- A Perplexity API key for dense embeddings
- Qdrant available either:
  - locally via `COSMIC_MEMORY_QDRANT_PATH`, or
  - remotely via `COSMIC_MEMORY_QDRANT_URL`

## Install

Production-style install:

```bash
python -m pip install -r requirements.txt
```

Development install:

```bash
python -m pip install -e .[dev]
```

Development install with the Neo4j graph backend:

```bash
python -m pip install -e .[dev,graph]
```

Development install with the local-path Qdrant BM25 stack:

```bash
python -m pip install -e .[dev,qdrant-local]
```

## Environment

Create a local `.env` file in the repo root or export variables directly.

Required:

```env
PERPLEXITY_API_KEY=your_key_here
```

Common optional variables:

```env
COSMIC_MEMORY_DATA_DIR=.cosmic-memory-data
COSMIC_MEMORY_EMBEDDING_MODEL=pplx-embed-v1-4b
COSMIC_MEMORY_EMBEDDING_DIMENSIONS=1024
COSMIC_MEMORY_EMBED_BATCH_SIZE=128
COSMIC_MEMORY_EMBED_MAX_PARALLEL=4
COSMIC_MEMORY_EMBED_ENCODING=base64_int8
COSMIC_MEMORY_QDRANT_COLLECTION=memories
COSMIC_MEMORY_QDRANT_PATH=.cosmic-memory-data/qdrant_data
COSMIC_MEMORY_SPARSE_MODEL=Qdrant/bm25
COSMIC_MEMORY_SPARSE_BACKEND=auto
COSMIC_MEMORY_PASSIVE_GRAPH_TIMEOUT_MS=120
COSMIC_MEMORY_SYNC_ON_STARTUP=true
```

If you use a remote Qdrant instance, set:

```env
COSMIC_MEMORY_QDRANT_URL=http://your-qdrant-host:6333
```

instead of `COSMIC_MEMORY_QDRANT_PATH`.

If you use the Neo4j graph backend, also set:

```env
COSMIC_MEMORY_GRAPH_BACKEND=neo4j
COSMIC_MEMORY_NEO4J_URI=bolt://127.0.0.1:7687
COSMIC_MEMORY_NEO4J_USERNAME=neo4j
COSMIC_MEMORY_NEO4J_PASSWORD=your_password
COSMIC_MEMORY_NEO4J_DATABASE=neo4j
```

## Qdrant Notes

- Production defaults to Qdrant-native BM25 for sparse retrieval.
- `COSMIC_MEMORY_SPARSE_BACKEND=auto` uses FastEmbed sparse encoding for local-path Qdrant and keeps native BM25 for remote Qdrant.
- This requires `qdrant-client>=1.15.2`.
- If you use `COSMIC_MEMORY_QDRANT_PATH` with native BM25, install `fastembed` too.
- Dense embeddings still come from Perplexity.
- Canonical Markdown remains the source of truth; Qdrant is a rebuildable index.

## Run

Production-oriented app:

```bash
python -m uvicorn cosmic_memory.server.app:create_default_app --factory --host 0.0.0.0 --port 8000
```

Explicit filesystem app:

```bash
python -m uvicorn cosmic_memory.server.app:create_filesystem_app --factory --host 0.0.0.0 --port 8000
```

Development-only fallback app:

```bash
python -m uvicorn cosmic_memory.server.app:create_development_app --factory --host 127.0.0.1 --port 8000
```

## Important Endpoints

- `GET /health`
- `POST /v1/embeddings/generate`
- `POST /v1/memories`
- `GET /v1/memories/{memory_id}`
- `POST /v1/memories/{memory_id}/supersede`
- `POST /v1/core-facts`
- `GET /v1/core-facts`
- `POST /v1/query/passive`
- `POST /v1/query/active`
- `GET /v1/index/status`
- `POST /v1/index/sync`
- `POST /v1/index/rebuild`

## Verification

Basic compile check:

```bash
python -m compileall -q src
```

Tests:

```bash
python -m pytest -q
```

Passive recall benchmark:

```bash
python benchmarks/passive_recall_benchmark.py --mode inprocess
```

HTTP-path benchmark:

```bash
python benchmarks/passive_recall_benchmark.py --mode http --base-url http://127.0.0.1:8000
```

## Current Scope

This repo currently provides:

- canonical Markdown memory records
- SQLite registry
- first-class `core_fact`
- Qdrant passive recall
- index sync and rebuild

It does not yet provide the full Gateway-side compaction and session summarization loop.
