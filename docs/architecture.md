# Cosmic Memory Architecture

## Purpose

`cosmic-memory` is the dedicated long-term memory layer for Cosmic.

It exists to separate memory concerns from the rest of the backend while still
supporting two integration modes:

- in-process library calls for low-latency hot paths,
- internal HTTP endpoints for orchestrator and agent access.

The current thin Cosmic Gateway integration uses the internal HTTP mode. Gateway
owns live session state in `sessions.db`, then calls `cosmic-memory` over
loopback HTTP for:

- `core_fact` block loading,
- passive recall during prompt assembly,
- transcript/episode ingestion for completed conversation turns,
- internal memory proxy routes for future `MemoryRead` / `MemoryWrite` tools.

That integration is intentionally thin for now:

- long-term memory is injected into direct-route and orchestrator prompts,
- the model-router also receives the assembled long-term memory block during classification,
- daily rollover summaries are now written back into long-term memory through the Gateway rollover path,
- task summaries and agent-note sync are still the next Gateway-side steps.

The app-factory split is now explicit:

- `create_default_app()` is production-oriented and env-backed,
- `create_filesystem_app()` is the same production path with a configurable data root,
- `create_development_app()` is the explicit in-memory development/test path.

## Boundary

This repository owns:

- canonical long-term memory records,
- runtime observation ingestion into canonical episode/transcript records,
- `memory_id` lifecycle,
- passive recall contracts,
- active memory search and traversal contracts,
- provenance and supersession rules,
- memory-layer service APIs.

This repository does not own:

- live daily session storage,
- sticky routing,
- current-day conversation replay,
- channel handling,
- task orchestration.

Those stay in the Cosmic Gateway and orchestrator.

## Memory Model

The target Cosmic model has three layers:

### 1. `core_fact`

Small, deterministic, always-on memory:

- identity,
- standing preferences,
- key relationships,
- long-lived goals,
- important constraints.

This layer is loaded directly, not semantically searched.

The current implementation in this repo now treats `core_fact` as a
first-class write path with:

- dedicated request models,
- deterministic prompt-block rendering,
- automatic supersession by `canonical_key`.

### 2. Passive Recall

Used on every query:

- query -> dense/sparse retrieval,
- fixed token budget,
- returns the highest-value long-term memories for prompt assembly.

Expected canonical sources:

- `core_fact`
- `session_summary`
- `task_summary`
- `agent_note`
- `user_data`

### 3. Active Agentic Memory

Used when Cosmic needs deep recall:

- query rewrite,
- multi-pass retrieval,
- graph traversal,
- contradiction inspection,
- structured memory brief generation.

This is where Graphiti-style ideas are relevant.

The current codebase now includes the first graph layer needed to support this:

- a typed ontology,
- deterministic identity-key normalization,
- strict merge rules for strong vs weak identity evidence,
- exact-name auto-merge for non-person entities such as projects and tasks,
- entity-level vector candidate generation over canonical names, aliases, and attributes,
- write-time LLM extraction into canonical `graph_document` payloads,
- internal LLM adjudication for ambiguous entity creation and merging,
- first-class graph episodes carrying source excerpt, timestamps, extraction confidence, and relation lineage,
- structured active-fact lookup over entity ids, compatible relation families, active-state filters, and time windows,
- on-ingest fact invalidation for ambiguous current-state conflicts,
- internal graph retrieval recipes for passive and active graph-backed recall,
- a first-class observation ingestion path for transcript/episode records,
- one-hop graph-assisted passive recall,
- graph-first active traversal when a graph store is attached.

The passive path is intentionally Qdrant-first and graph-assisted, not
graph-dominant:

- bounded hybrid candidate retrieval from the passive index,
- query-aware reranking using type, recency, identity, current-state, and token-cost signals,
- strict token-budget packing before prompt assembly,
- optional entity-similarity lookup for indirect graph seed generation,
- shallow graph assist only when the query looks relationship-heavy,
- passive graph hits are post-processed through a cheap recipe layer using RRF, node-distance, and episode-mention signals,
- bounded graph wait time so passive recall does not stall on traversal work.

The active path now has its own recipe layer:

- graph traversal produces candidate relations, entities, episodes, and supporting memories,
- an internal active recipe reranks those relations with:
  - lexical/query relevance
  - current-vs-historical state alignment
  - node distance from seeded entities
  - episode support and recency
  - MMR diversification to avoid repeated near-duplicate relations
- supporting memory ids are then ranked from the selected relation set before canonical record hydration

The recipe layer is intentionally hybrid rather than pure-RRF:

- weighted signal scores remain the main driver for interpretability,
- RRF now has explicit weight and scale so it contributes materially instead of acting as a decorative tiebreaker,
- MMR is only used on active traversal, not on passive graph assist.

## Storage Direction

The planned storage split is:

- canonical truth:
  - Markdown records
  - SQLite registry
- passive index:
  - Qdrant
  - dense embeddings generated by an isolated embedding service
  - sparse retrieval via Qdrant-native BM25 by default
- entity similarity index:
  - separate Qdrant collection
  - same Perplexity embedding service as passive memory
  - used for entity merge candidates and graph query seeding
  - in embedded local mode, uses a separate local Qdrant path from the passive index to avoid client lock conflicts
- active traversal projection:
  - graph backend

The graph is a projection, not the primary source of truth.

When graph extraction is enabled, canonical write flow becomes:

1. write request enters the memory service,
2. an LLM extracts entities, identity candidates, aliases, typed relations, and temporal fields,
3. the extraction result is deduplicated and normalized locally,
4. the memory layer creates or updates a first-class graph episode for provenance,
5. ambiguous entity candidates can be passed to an internal adjudicator,
6. candidate active facts are retrieved from the graph using a structured fact query,
7. ambiguous fact conflicts can be passed to an internal fact adjudicator,
8. the normalized `graph_document` is written back into canonical Markdown metadata,
9. the same normalized document is ingested into the graph projection,
10. changed entities are synced into the entity-similarity index,
11. the passive index is updated independently.

This keeps graph intelligence on the ingestion side and keeps passive recall
free of query-time LLM latency.

There is now also a runtime observation-ingestion path:

- Gateway/orchestrator can send one or more conversation/task observations through `ingest_episode(...)`
- `cosmic-memory` renders them into a canonical transcript-style record
- transcript ingestion can opt into graph extraction explicitly, even though transcript memories are not extracted by default
- the same canonical-write, graph-ingest, and passive-index sync path is then reused

This is the missing bridge between live Cosmic runtime events and the long-term
memory subsystem.

## Episode And Fact Model

Graphiti-style provenance is now treated as a first-class requirement rather
than an implementation detail.

Each ingested graph document can produce:

- one `episode`
  - `episode_id`
  - `memory_id`
  - source type and provenance identifiers
  - `created_at`
  - `extracted_at`
  - extraction confidence
  - source excerpt
  - produced relation ids
  - invalidated relation ids
- one or more graph relations
  - each relation stores its contributing `episode_ids`
  - each invalidated relation stores `invalidated_by_episode_id`

This is important because invalidation is not just a graph mutation. It is a
lineage event: a later episode caused an earlier active fact to stop being
current.

## Structured Fact Lookup

Fact invalidation depends on a dedicated graph-store primitive rather than
generic traversal plus client-side filtering.

The graph store now exposes a structured fact query that can filter by:

- anchor entity ids
- explicit source entity ids
- explicit target entity ids
- compatible relation families
- active-only state
- optional time window

That query is used in two places:

- ingestion-time invalidation candidate retrieval
- future agent/orchestrator current-state and temporal tools

This keeps the invalidation pipeline attached to the right abstraction and
makes it easier to optimize both the in-memory and Neo4j backends.

## Invalidation Model

Fact invalidation is primarily a write-time concern, not a cron concern.

When a new graph fact is ingested:

1. deterministic duplicate checks run first
2. active compatible facts are retrieved through the structured fact query
3. if needed, an internal fact adjudicator reasons over the pending fact, the
   active candidates, their episodes, and the time anchors
4. the system decides whether to:
   - keep both
   - merge with an existing relation
   - invalidate one or more existing active relations
   - discard the new fact
5. invalidated relations are marked with `invalidated_by_episode_id`
6. current-state traversal excludes invalidated relations by default

Periodic consolidation is still useful later, but the main correctness path is
on ingestion so the graph does not remain stale between writes.

There is also a deterministic fallback before the LLM adjudicator:

- if the pending fact is a later update for the same source entity, target
  entity, and relation type,
- and the earlier fact is still active,
- and the fact text is not an exact duplicate,

then the older relation can be invalidated immediately without paying LLM
latency.

## Query Semantics

The graph fact query is intentionally explicit:

- `source_entity_ids`
  - directional filter on the relation source side only
- `target_entity_ids`
  - directional filter on the relation target side only
- `anchor_entity_ids`
  - undirected anchor filter; a relation matches if the entity appears on either side

Traversal also has an explicit current-vs-historical split:

- `prefer_current_state=true`
  - favor active/current facts and suppress invalidated relations where possible
- `prefer_current_state=false`
  - allow broader historical traversal, including invalidated relations when relevant
- `relation_distances`
  - 1-indexed hop counts from the nearest seed entity
  - direct seed-adjacent relations have distance `1`

## Retrieval Recipes

This repo now has an internal recipe layer inspired by Graphiti's search
recipes, but kept intentionally cheaper and narrower for Cosmic's current
stage.

The implemented recipes today are:

- passive graph assist
  - `passive_hybrid_rrf`
  - uses a weighted hybrid of lexical relevance, state alignment, node distance, episode support, and materially scaled RRF
  - no MMR because passive recall is latency-sensitive and only needs a bounded boost set
- active current-state traversal
  - `active_current_state_rrf_mmr`
  - uses the same hybrid scoring plus MMR diversification
- active temporal traversal
  - `active_temporal_rrf_mmr`
  - biases more toward episode support and historical state
- active generic traversal
  - `active_hybrid_rrf_mmr`

These recipes are internal. They are not public API yet. The service layer
selects them automatically from the query frame and uses them to reorder graph
results and supporting canonical memories before building the final response.

Entity candidate resolution is intentionally layered:

1. exact strong identity keys such as email/phone/external account
2. exact normalized-name auto-merge for allowed non-person entity types
3. weak alias-only matches become provisional candidates
4. entity vector similarity produces a shortlist over compatible ontology types
5. an internal adjudicator can reason over shortlisted candidates for ambiguous cases
6. only very high-confidence similarity hits auto-merge without adjudication; the rest stay provisional

At the moment, the implemented graph backends are intentionally limited:

- `InMemoryGraphStore` for development and local validation
- `Neo4jGraphStore` as the first persistent backend boundary

This is deliberate. The persistent backend should be added only after the
ontology, normalization, and traversal contracts are stable.

The current implemented storage path already follows the Cosmic architecture
pattern:

1. write canonical Markdown record,
2. update the SQLite registry,
3. optionally sync the passive retrieval index.

That keeps `.md` as source of truth and treats the passive index as rebuildable.

The current passive index implementation also keeps dense embedding generation
isolated from storage:

- the embedding service lives in `src/cosmic_memory/embeddings/`,
- production app factories load environment from a local `.env` file when present,
- production factories require a real `PERPLEXITY_API_KEY` instead of silently
  falling back to the development embedder,
- the server exposes `/v1/embeddings/generate` as a separate endpoint,
- Qdrant sync/search calls the same embedding service instead of generating
  vectors inline,
- native BM25 text conversion happens inside Qdrant by default,
- the app's `auto` sparse mode uses FastEmbed for local-path Qdrant and keeps
  native BM25 for remote Qdrant,
- local-path Qdrant still requires `fastembed` in the client environment for
  native document/sparse inference,
- explicit sparse encoders remain available as a fallback for tests and older clients.

The passive index now also includes an explicit control plane:

- `GET /v1/index/status`
  - compares canonical Markdown, registry state, and Qdrant state
  - reports missing, stale, and orphaned records
- `POST /v1/index/sync`
  - repairs registry drift from canonical files
  - upserts missing/stale Qdrant points
  - removes orphaned Qdrant points
- `POST /v1/index/rebuild`
  - resets and repopulates the Qdrant collection from canonical records

This mirrors the current Cosmic architecture direction:

- Markdown is the source of truth,
- the registry is rebuildable from Markdown,
- Qdrant is rebuildable from canonical records,
- startup sync is allowed as a safety mechanism.

## Service Shape

The memory layer should expose:

- write memory,
- get memory,
- supersede memory,
- generate embeddings,
- passive recall,
- active recall,
- agent-facing memory control primitives:
  - schema context,
  - query planning,
  - identity resolution,
  - current-state lookup,
  - temporal fact lookup,
  - structured memory briefs,
- health / rebuild / maintenance endpoints.

These are not meant to be arbitrary helper endpoints. They are the typed memory
control surface the orchestrator should use instead of generating raw Cypher or
re-implementing memory heuristics in every agent.

The same principle applies to memory authoring: agents should use a small
number of canonical write paths, while entity-candidate search and ambiguous
merge decisions stay inside the memory subsystem as internal adjudication work.

## Why Library-First

Passive recall is on Cosmic's hot path. If the Gateway and memory layer are
running together, the Gateway should be able to call the memory core directly
without forcing a network hop.

The HTTP server still matters because:

- agents and orchestrator need stable internal APIs,
- local and remote deployment modes should both be possible,
- the memory layer should have a clean service boundary.

## What We Want to Borrow

From ReMe:

- compaction and working-memory hygiene,
- practical recall ergonomics,
- contradiction/redundancy filtering patterns.

From Graphiti:

- episode-based ingestion,
- typed entity and fact extraction,
- temporal fact handling,
- graph-aware retrieval and traversal.

## Current Milestone

The current milestone in this repo now includes:

- define the contracts,
- provide a development implementation,
- provide first-class `core_fact` operations,
- provide canonical Markdown record storage,
- provide a memory-owned SQLite registry,
- provide an isolated dense embedding subsystem,
- provide Perplexity standard embedding support,
- provide a filesystem-backed service implementation,
- provide a server wrapper,
- provide a Qdrant-native passive-recall adapter boundary,
- provide startup and on-demand index sync/rebuild workflows,
- provide token-budget-aware passive recall with multi-factor reranking and budget packing,
- provide batch and parallel request handling for embedding/index work.
- provide a dedicated entity-similarity index for merge candidate generation and graph seeding,
- provide an internal xAI-backed adjudicator for ambiguous entity resolution on write,
- provide an agent/orchestrator control surface for memory planning and graph-aware lookup.

The next backend milestones are:

- Gateway integration against the real session assembly path,
- consolidation workflows.
