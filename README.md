# cosmic-memory

Memory layer for Cosmic.

This repository is the foundation for Cosmic's long-term memory system. It is
being built as a library-first Python package with an optional API server so
Cosmic can:

- keep memory logic isolated from the Gateway and agents,
- preserve fast in-process access for hot-path passive recall,
- expose stable internal endpoints for orchestrator and agent use,
- evolve the memory layer independently over time.

## Design Direction

The intended Cosmic memory stack is:

- `core_fact`
  - always-on user profile and standing preferences
- passive recall
  - fast retrieval for every query
  - canonical records -> Qdrant hybrid retrieval
- active agentic memory
  - deep search, traversal, contradiction handling, graph projection

This repo is not intended to own Cosmic's live message/session routing. The
Gateway still owns the current-day session store and conversation orchestration.
`cosmic-memory` owns canonical long-term memory, retrieval contracts, and
memory-layer operations.

## Current Scope

The current milestone in this repo provides:

- project scaffolding,
- core memory schemas and service interfaces,
- first-class `core_fact` writes and deterministic profile block rendering,
- canonical Markdown record storage,
- a memory-owned SQLite registry,
- a filesystem-backed canonical memory service,
- an isolated embedding subsystem,
- a dedicated `/v1/embeddings/generate` endpoint,
- a Qdrant passive-recall adapter using native BM25 by default,
- startup and on-demand index sync/rebuild operations,
- token-budget-aware passive recall with multi-factor reranking,
- graph ontology and deterministic identity normalization foundations,
- write-time xAI graph extraction with structured output support,
- document-level graph dedup normalization before graph ingest,
- graph-assisted passive recall and graph-first active recall when a graph store is attached,
- a compact agent/orchestrator memory control surface,
- schema injection, query planning, identity resolution, current-state lookup, temporal fact lookup, and structured memory briefs,
- a thin FastAPI server,
- an in-memory development implementation for contract testing.

That gives us a real shape for the system before wiring in:

- consolidation and supersession jobs.

## Repository Layout

```text
src/cosmic_memory/
  control_surface.py # agent/orchestrator-facing memory control surface
  core_facts.py # deterministic always-on profile block helpers
  domain/      # schemas, enums, contracts
  embeddings/  # dense embedding services
  index/       # passive recall index adapters
  storage/     # canonical markdown + sqlite registry
  server/      # FastAPI wrapper
  service.py   # abstract memory service contract
  dev_service.py
  filesystem_service.py
docs/
  architecture.md
tests/
```

## Quick Start

```bash
python -m pip install -e .[dev]
python -m uvicorn cosmic_memory.server.app:create_default_app --factory --reload
pytest -q
```

Production behavior:

- `create_default_app()` is production-oriented and env-backed.
- `create_filesystem_app()` is the same production path with an optional custom
  data directory.
- `create_development_app()` is the only deterministic local fallback path.

`create_default_app()` and `create_filesystem_app()` require
`PERPLEXITY_API_KEY` and use Perplexity standard embeddings with
`pplx-embed-v1-4b`.

If `XAI_API_KEY` is present, production app factories also enable write-time
graph extraction with `grok-4-1-fast-reasoning` by default.

Production app factories also load a local `.env` file if present. A placeholder
is included in [.env.example](C:/Users/Praveen Raj U S/Downloads/cosmic-memory/.env.example).

Relevant environment variables:

- `PERPLEXITY_API_KEY`
- `XAI_API_KEY`
- `COSMIC_MEMORY_EMBEDDING_MODEL` (default `pplx-embed-v1-4b`)
- `COSMIC_MEMORY_EMBEDDING_DIMENSIONS` (default `1024`)
- `COSMIC_MEMORY_EMBED_BATCH_SIZE` (default `128`)
- `COSMIC_MEMORY_EMBED_MAX_PARALLEL` (default `4`)
- `COSMIC_MEMORY_EMBED_ENCODING` (default `base64_int8`)
- `COSMIC_MEMORY_QDRANT_URL` or `COSMIC_MEMORY_QDRANT_PATH`
- `COSMIC_MEMORY_QDRANT_COLLECTION` (default `memories`)
- `COSMIC_MEMORY_SPARSE_MODEL` (default `Qdrant/bm25`)
- `COSMIC_MEMORY_SPARSE_BACKEND` (`auto`, `native`, `fastembed`, or `simple`; default `auto`)
- `COSMIC_MEMORY_PASSIVE_GRAPH_TIMEOUT_MS` (default `120`)
- `COSMIC_MEMORY_SYNC_ON_STARTUP` or `MEMORY_SYNC_ON_STARTUP` (default `true`)
- `COSMIC_MEMORY_GRAPH_BACKEND` (`none`, `memory`, or `neo4j`, default `none`)
- `COSMIC_MEMORY_NEO4J_URI`
- `COSMIC_MEMORY_NEO4J_USERNAME`
- `COSMIC_MEMORY_NEO4J_PASSWORD`
- `COSMIC_MEMORY_NEO4J_DATABASE` (default `neo4j`)
- `COSMIC_MEMORY_GRAPH_EXTRACT_ENABLED` (default `true` when `XAI_API_KEY` is present)
- `COSMIC_MEMORY_GRAPH_EXTRACT_MODEL` (default `grok-4-1-fast-reasoning`)
- `COSMIC_MEMORY_GRAPH_EXTRACT_MAX_PARALLEL` (default `2`)
- `COSMIC_MEMORY_GRAPH_EXTRACT_MAX_RETRIES` (default `3`)
- `COSMIC_MEMORY_GRAPH_EXTRACT_RETRY_BASE_SECONDS` (default `1.0`)
- `COSMIC_MEMORY_GRAPH_EXTRACT_RETRY_MAX_SECONDS` (default `12.0`)
- `COSMIC_MEMORY_TIMEZONE` (default `UTC`)
- `COSMIC_MEMORY_ENV_FILE` (default `.env`)

Passive retrieval notes:

- dense vectors come from Perplexity embeddings,
- sparse retrieval defaults to Qdrant-native BM25,
- `COSMIC_MEMORY_SPARSE_BACKEND=auto` uses FastEmbed sparse encoding for local-path Qdrant and keeps native BM25 for remote/server Qdrant,
- `qdrant-client>=1.15.2` is required for the native BM25 path,
- local-path Qdrant with native BM25 also requires `fastembed` in the client environment,
- explicit sparse encoders are still supported for tests and older environments.
- passive recall overfetches a bounded candidate window, reranks by lexical/index score, type, recency, exact identity hits, current-state hints, graph support, and token cost,
- final selection packs the token budget instead of blindly taking the biggest top hit,
- graph assistance is shallow and bounded by timeout so it can improve recall without owning the hot path.

Graph notes:

- identity keys are deterministic and normalized before hashing,
- exact strong keys such as email auto-link entities,
- weak alias keys do not auto-merge by themselves,
- non-person entities such as `project`, `task`, and `organization` can auto-merge by exact normalized name,
- write-time extraction stores normalized `graph_document` payloads back into canonical Markdown,
- extraction prompts are explicitly time-aware and resolve relative dates against UTC and local timezone anchors,
- current graph backend support in the app factory is intentionally limited to:
  - `memory` for local/dev validation
  - `neo4j` as the first persistent backend boundary
  - `none` when graph traversal is disabled

Benchmarking:

- `python benchmarks/passive_recall_benchmark.py --mode inprocess`
- the benchmark reports `p50`, `p80`, `p95`, top-1 hit rate, any-hit rate, and token-budget compliance
- `--mode http --base-url http://127.0.0.1:8000` exercises the HTTP boundary end to end

Current server endpoints:

- `GET /health`
- `POST /v1/embeddings/generate`
- `POST /v1/memories`
- `GET /v1/memories/{memory_id}`
- `POST /v1/memories/{memory_id}/supersede`
- `POST /v1/core-facts`
- `GET /v1/core-facts`
- `POST /v1/query/passive`
- `POST /v1/query/active`
- `GET /v1/agent/schema-context`
- `POST /v1/agent/plan`
- `POST /v1/agent/resolve-identity`
- `POST /v1/agent/current-state`
- `POST /v1/agent/temporal-facts`
- `POST /v1/agent/memory-brief`
- `GET /v1/index/status`
- `POST /v1/index/sync`
- `POST /v1/index/rebuild`

Index behavior is aligned with the current Cosmic architecture:

- canonical Markdown remains the source of truth,
- the SQLite registry is a fast lookup/cache layer for canonical records,
- Qdrant is treated as a rebuildable passive index,
- startup sync can repair registry/index drift from canonical files,
- passive recall enforces a fixed token budget before returning memories.

## Agent Control Surface

The agent-facing API is intentionally small. It is not meant to expose raw graph
or index internals; it is meant to give the orchestrator stable memory
primitives:

- `schema_context`
  - inject ontology, memory kinds, relation types, and tool guidance before complex planning
- `plan`
  - convert a user/task query into a memory search plan with recommended mode and tool sequence
- `resolve_identity`
  - normalize exact keys like email or username before traversal
- `current_state`
  - return active facts such as blockers, owners, reminders, preferences, and current work
- `temporal_facts`
  - return time-aware facts for before/after/when/history questions
- `memory_brief`
  - bundle passive recall, active recall, current/temporal facts, and distilled findings for the orchestrator

This keeps memory reasoning inside `cosmic-memory` instead of forcing every
agent to relearn the ontology, traversal strategy, and ranking heuristics.

## Near-Term Plan

1. Integrate passive recall with Cosmic Gateway session assembly.
2. Wire the new control surface into Gateway and orchestrator memory tooling.
3. Add consolidation and conflict-resolution jobs.
4. Add extraction queues, caches, and production observability.
