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

## Architecture

### System Boundaries

```mermaid
flowchart LR
    U[User Query] --> G[Cosmic Gateway]
    G --> S[sessions.db<br/>live daily session]
    G --> CM[cosmic-memory]

    subgraph CMStack[cosmic-memory]
        CF[core_fact block]
        PR[Passive Recall]
        AR[Active Agentic Memory]
        CS[Agent Control Surface]
        EXT[LLM Graph Extraction]
        ADJ[Internal Memory Adjudicator]
        REG[SQLite Registry]
        MD[Canonical Markdown]
        Q[(Qdrant<br/>memory index)]
        EQ[(Qdrant<br/>entity index)]
        N[(Neo4j<br/>graph projection)]
    end

    CM --> CF
    CM --> PR
    CM --> AR
    CM --> CS
    EXT --> MD
    ADJ --> N
    MD <--> REG
    MD --> Q
    MD --> N
    N --> EQ
    PR --> Q
    PR --> N
    PR --> EQ
    AR --> Q
    AR --> N

    CF --> G
    PR --> G
    CS --> G
```

### Write And Ingestion Flow

```mermaid
sequenceDiagram
    participant GW as Gateway / Agent
    participant CM as cosmic-memory
    participant X as xAI Extractor
    participant A as Memory Adjudicator
    participant MD as Markdown Store
    participant REG as SQLite Registry
    participant G as Graph Store
    participant EI as Entity Index
    participant Q as Qdrant Memory Index

    GW->>CM: write(memory)
    CM->>MD: persist canonical .md record
    CM->>REG: upsert registry snapshot
    alt graph extraction enabled
        CM->>X: extract entities / relations / time anchors
        X-->>CM: graph_document
        CM->>MD: write normalized graph metadata
        CM->>A: adjudicate ambiguous entity merges
        A-->>CM: merge / candidate / create_new
        CM->>G: ingest graph_document
        G->>EI: sync changed entities
    end
    CM->>Q: upsert passive memory point
    CM-->>GW: MemoryRecord + memory_id
```

### Passive And Active Retrieval

```mermaid
flowchart TD
    QRY[Incoming query] --> CORE[Load core_fact block]
    QRY --> PASS[Passive recall]

    PASS --> MEMQ[Qdrant hybrid memory search]
    PASS --> RERANK[Multi-factor reranking<br/>type + recency + identity + current-state + token cost]
    PASS --> GSEED[Optional graph seed generation<br/>exact identities + entity similarity]
    GSEED --> G1[Shallow graph assist<br/>1-hop, timeout bounded]
    MEMQ --> RERANK
    G1 --> RERANK
    RERANK --> PACK[Token-budget packing]
    PACK --> CTX[Prompt context for every LLM call]

    QRY --> DECIDE{Need deep memory?}
    DECIDE -->|Yes| ACTIVE[Active memory traversal]
    ACTIVE --> PLAN[Agent/orchestrator memory plan]
    PLAN --> TOOLS[resolve_identity / current_state / temporal_facts / traverse / memory_brief]
    TOOLS --> GRAPH[Multi-hop graph + memory evidence]
    GRAPH --> BRIEF[Structured memory brief]
    DECIDE -->|No| CTX
```

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
- internal xAI-backed entity adjudication for ambiguous graph writes,
- document-level graph dedup normalization before graph ingest,
- a dedicated entity-similarity index for entity candidate generation and graph seeding,
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
  extraction/  # graph extraction and normalization
  graph/       # ontology, identity resolution, traversal, entity similarity
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
- `COSMIC_MEMORY_ENTITY_INDEX_ENABLED` (default `true`)
- `COSMIC_MEMORY_ENTITY_COLLECTION` (default `memory_entities`)
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
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_ENABLED` (default `true` when `XAI_API_KEY` is present)
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_MODEL` (default `grok-4-1-fast-reasoning`)
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_MAX_PARALLEL` (default `2`)
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_MAX_RETRIES` (default `3`)
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_RETRY_BASE_SECONDS` (default `1.0`)
- `COSMIC_MEMORY_GRAPH_ADJUDICATE_RETRY_MAX_SECONDS` (default `12.0`)
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
- entity similarity uses a separate Qdrant collection and the same Perplexity embedding model as passive memory,
- vector similarity is only used for shortlist generation after exact identity and normalized-name checks,
- ambiguous entity creation and merge decisions can be escalated to an internal xAI adjudicator,
- ambiguous similarity hits become provisional `candidate_match` results instead of silent auto-merges,
- write-time extraction stores normalized `graph_document` payloads back into canonical Markdown,
- extraction prompts are explicitly time-aware and resolve relative dates against UTC and local timezone anchors,
- passive graph seeding can use the entity-similarity index to resolve indirect references like `todo` -> `task`,
- current graph backend support in the app factory is intentionally limited to:
  - `memory` for local/dev validation
  - `neo4j` as the first persistent backend boundary
  - `none` when graph traversal is disabled

Benchmarking:

- `python benchmarks/passive_recall_benchmark.py --mode inprocess`
- the benchmark reports `p50`, `p80`, `p95`, top-1 hit rate, any-hit rate, and token-budget compliance
- `--mode http --base-url http://127.0.0.1:8000` exercises the HTTP boundary end to end

Current server endpoints:

| Method | Path | Purpose | Used By | Notes |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | Health/status check for the service. | Infra, local dev, Gateway | Not memory-specific. |
| `POST` | `/v1/embeddings/generate` | Generate dense embeddings through the configured embedding backend. | Internal tooling, diagnostics, future batch jobs | Not on the normal recall hot path. |
| `POST` | `/v1/memories` | Write a canonical long-term memory record. | Gateway, agents, orchestrator | Canonical `.md` write path. |
| `GET` | `/v1/memories/{memory_id}` | Fetch one canonical memory by ID. | Gateway, agents, orchestrator | Direct record lookup. |
| `POST` | `/v1/memories/{memory_id}/supersede` | Supersede an existing memory with a replacement record. | Gateway, agents, consolidation jobs | Preserves history instead of blind overwrite. |
| `POST` | `/v1/core-facts` | Write a first-class `core_fact` record. | Gateway, profile updates, agents | Supports `canonical_key`-based supersession. |
| `GET` | `/v1/core-facts` | Build the deterministic always-on core profile block. | Gateway | Hot prompt-assembly dependency. |
| `POST` | `/v1/query/passive` | Fast token-budgeted passive recall for every query. | Gateway | Hot path. Qdrant-first, graph-assisted. |
| `POST` | `/v1/query/active` | Deep memory search and graph-backed active recall. | Orchestrator, agents | Not hot path. Used for harder memory tasks. |
| `GET` | `/v1/agent/schema-context` | Return ontology, memory kinds, and tool guidance for planning. | Orchestrator, agents | Schema injection surface. |
| `POST` | `/v1/agent/plan` | Convert a query into a memory search plan. | Orchestrator, agents | Planning helper, not raw DB access. |
| `POST` | `/v1/agent/resolve-identity` | Normalize and resolve identity keys such as email or username. | Orchestrator, agents | Exact identity tool. |
| `POST` | `/v1/agent/current-state` | Return current facts such as blockers, reminders, ownership, or active work. | Orchestrator, agents | Current-state oriented graph lookup. |
| `POST` | `/v1/agent/temporal-facts` | Return historical or time-bounded facts. | Orchestrator, agents | For before/after/when questions. |
| `POST` | `/v1/agent/memory-brief` | Build a structured memory brief from passive + active evidence. | Orchestrator, agents | High-level agent-facing synthesis tool. |
| `GET` | `/v1/index/status` | Report canonical/registry/index consistency state. | Ops, local dev, maintenance jobs | Control-plane endpoint. |
| `POST` | `/v1/index/sync` | Repair drift between canonical files, registry, and passive index. | Ops, startup hooks, maintenance jobs | Incremental repair path. |
| `POST` | `/v1/index/rebuild` | Rebuild the passive index from canonical records. | Ops, maintenance jobs | Full rebuild path. |

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

Memory authoring intentionally follows the same rule: agents are expected to
call high-level canonical write paths, while entity-candidate retrieval and
ambiguous merge decisions stay inside `cosmic-memory` as an internal
adjudication step.

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
