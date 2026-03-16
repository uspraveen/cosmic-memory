"""Experimental persistent Neo4j graph store."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
from datetime import datetime
import logging
from time import perf_counter

from cosmic_memory.domain.models import GraphStoreStats, utc_now
from cosmic_memory.graph.dev_store import InMemoryGraphStore
from cosmic_memory.graph.adjudication import (
    EntityAdjudicationRequest,
    EntityAdjudicationService,
    EntityCandidateContext,
)
from cosmic_memory.graph.fact_adjudication import (
    FactAdjudicationRequest,
    FactAdjudicationService,
    FactCandidateContext,
    PendingFactContext,
    exact_fact_signature,
)
from cosmic_memory.graph.entity_index import EntitySimilarityIndex, entity_similarity_text_parts
from cosmic_memory.graph.identity import build_identity_key
from cosmic_memory.graph.models import (
    GraphDocument,
    GraphEntityNode,
    GraphEpisode,
    GraphFactQuery,
    GraphIdentityCandidate,
    GraphIdentityKey,
    GraphIngestResult,
    GraphRelationEdge,
    GraphSearchResult,
    IdentityResolutionCandidate,
    IdentityResolutionResult,
)
from cosmic_memory.graph.ontology import IdentityKeyType, RelationType, compatible_relation_types
from cosmic_memory.graph.resolution import (
    STRONG_KEY_TYPES,
    entity_allows_name_auto_merge,
    similarity_candidate_types,
)
from cosmic_memory.graph.xai_adjudicator import build_local_time_anchor

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Persistent graph store backed by Neo4j.

    This v1 adapter keeps writes persistent in Neo4j and reuses the current
    in-memory scorer for passive/traversal search by hydrating a temporary
    search snapshot from the graph. That keeps the retrieval semantics aligned
    while the persistent backend contract stabilizes.
    """

    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        entity_index: EntitySimilarityIndex | None = None,
        adjudicator: EntityAdjudicationService | None = None,
        fact_adjudicator: FactAdjudicationService | None = None,
    ) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise ImportError(
                "neo4j is required for Neo4jGraphStore. "
                "Install project dependencies with `python -m pip install -e .[graph]`."
            ) from exc

        self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self.entity_index = entity_index
        self.adjudicator = adjudicator
        self.fact_adjudicator = fact_adjudicator
        self._ready = False
        self._cache_lock = asyncio.Lock()
        self._search_cache: InMemoryGraphStore | None = None
        self._cache_hydrated_at: datetime | None = None
        self._cache_build_ms: float | None = None

    async def ingest_document(self, document: GraphDocument) -> GraphIngestResult:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            ref_to_entity_id: dict[str, str] = {}
            resolution_events: list[IdentityResolutionResult] = []
            changed_entities: dict[str, GraphEntityNode] = {}
            episode = _coerce_episode(document)
            await self._persist_episode(session, episode)

            for entity in document.entities:
                resolved_entity, events = await self._upsert_entity(
                    session,
                    document=document,
                    document_entity=entity,
                )
                ref_to_entity_id[entity.local_ref] = resolved_entity.entity_id
                changed_entities[resolved_entity.entity_id] = resolved_entity
                resolution_events.extend(events)

            relation_ids: list[str] = []
            invalidated_relation_ids: list[str] = []
            for relation in document.relations:
                source_entity_id = ref_to_entity_id.get(relation.source_ref)
                target_entity_id = ref_to_entity_id.get(relation.target_ref)
                if source_entity_id is None or target_entity_id is None:
                    continue
                edge, invalidated_ids = await self._upsert_relation(
                    session,
                    document=document,
                    episode=episode,
                    memory_id=document.memory_id,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relation_type=relation.relation_type,
                    fact=relation.fact,
                    confidence=relation.confidence,
                    valid_at=relation.valid_at,
                    invalid_at=relation.invalid_at,
                    expires_at=relation.expires_at,
                )
                if edge is not None:
                    relation_ids.append(edge.relation_id)
                invalidated_relation_ids.extend(invalidated_ids)

            episode.produced_relation_ids = sorted(set(relation_ids))
            episode.invalidated_relation_ids = sorted(set(invalidated_relation_ids))
            await self._persist_episode(session, episode)

        result = GraphIngestResult(
            memory_id=document.memory_id,
            episode_id=episode.episode_id,
            entity_ids=list(dict.fromkeys(ref_to_entity_id.values())),
            relation_ids=list(dict.fromkeys(relation_ids)),
            invalidated_relation_ids=list(dict.fromkeys(invalidated_relation_ids)),
            resolution_events=resolution_events,
        )
        if self.entity_index is not None and changed_entities:
            await self.entity_index.sync_entities(list(changed_entities.values()))
        await self._cache_ingest_document(document)
        return result

    async def remove_memory(self, memory_id: str) -> None:
        await self._ensure_ready()
        touched_entity_ids: set[str] = set()
        episode_ids_to_remove: list[str] = []
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (ep:Episode {memory_id: $memory_id})
                RETURN ep.episode_id AS episode_id
                """,
                memory_id=memory_id,
            )
            async for row in result:
                episode_ids_to_remove.append(row["episode_id"])

            result = await session.run(
                """
                MATCH (rel:Relation)
                WHERE $memory_id IN coalesce(rel.memory_ids, [])
                RETURN rel.relation_id AS relation_id, rel.memory_ids AS memory_ids
                """,
                memory_id=memory_id,
            )
            async for row in result:
                remaining = [value for value in (row["memory_ids"] or []) if value != memory_id]
                if remaining:
                    await session.run(
                        """
                        MATCH (rel:Relation {relation_id: $relation_id})
                        SET rel.memory_ids = $memory_ids,
                            rel.episode_ids = [value IN coalesce(rel.episode_ids, []) WHERE NOT value IN $episode_ids],
                            rel.updated_at = $updated_at
                        """,
                        relation_id=row["relation_id"],
                        memory_ids=remaining,
                        episode_ids=episode_ids_to_remove,
                        updated_at=_serialize_datetime(utc_now()),
                    )
                else:
                    await session.run(
                        """
                        MATCH (rel:Relation {relation_id: $relation_id})
                        DETACH DELETE rel
                        """,
                        relation_id=row["relation_id"],
                    )

            if episode_ids_to_remove:
                await session.run(
                    """
                    MATCH (rel:Relation)
                    WHERE rel.invalidated_by_episode_id IN $episode_ids
                    SET rel.invalidated_by_episode_id = NULL,
                        rel.updated_at = $updated_at
                    """,
                    episode_ids=episode_ids_to_remove,
                    updated_at=_serialize_datetime(utc_now()),
                )

            for label in ("Entity", "IdentityKey"):
                id_field = "entity_id" if label == "Entity" else "key_id"
                result = await session.run(
                    f"""
                    MATCH (n:{label})
                    WHERE $memory_id IN coalesce(n.memory_ids, [])
                    RETURN n.{id_field} AS node_id, n.memory_ids AS memory_ids
                    """,
                    memory_id=memory_id,
                )
                async for row in result:
                    remaining = [value for value in (row["memory_ids"] or []) if value != memory_id]
                    if remaining:
                        await session.run(
                            f"""
                            MATCH (n:{label} {{{id_field}: $node_id}})
                            SET n.memory_ids = $memory_ids,
                                n.updated_at = $updated_at
                            """,
                            node_id=row["node_id"],
                            memory_ids=remaining,
                            updated_at=_serialize_datetime(utc_now()),
                        )
                    else:
                        await session.run(
                            f"""
                            MATCH (n:{label} {{{id_field}: $node_id}})
                            DETACH DELETE n
                            """,
                            node_id=row["node_id"],
                        )
                    if label == "Entity":
                        touched_entity_ids.add(row["node_id"])

            if episode_ids_to_remove:
                await session.run(
                    """
                    MATCH (ep:Episode)
                    WHERE ep.episode_id IN $episode_ids
                    DETACH DELETE ep
                    """,
                    episode_ids=episode_ids_to_remove,
                )
        await self._cache_remove_memory(memory_id)
        if self.entity_index is not None and touched_entity_ids:
            entities_to_sync: list[GraphEntityNode] = []
            entities_to_delete: list[str] = []
            for entity_id in touched_entity_ids:
                entity = await self.get_entity(entity_id)
                if entity is None or not entity.memory_ids:
                    entities_to_delete.append(entity_id)
                    continue
                entities_to_sync.append(entity)
            if entities_to_delete:
                await self.entity_index.delete_entities(entities_to_delete)
            if entities_to_sync:
                await self.entity_index.sync_entities(entities_to_sync)

    async def reset(self) -> None:
        await self._ensure_ready()
        entity_ids: list[str] = []
        if self.entity_index is not None:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("MATCH (e:Entity) RETURN e.entity_id AS entity_id")
                async for row in result:
                    entity_ids.append(row["entity_id"])
        async with self.driver.session(database=self.database) as session:
            await session.run("MATCH (n) DETACH DELETE n")
        if self.entity_index is not None and entity_ids:
            await self.entity_index.delete_entities(entity_ids)
        async with self._cache_lock:
            self._search_cache = None
            self._cache_hydrated_at = None
            self._cache_build_ms = None

    async def stats(self) -> GraphStoreStats:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            memory_result = await session.run("MATCH (ep:Episode) RETURN count(ep) AS count")
            entity_result = await session.run("MATCH (e:Entity) RETURN count(e) AS count")
            relation_result = await session.run("MATCH (r:Relation) RETURN count(r) AS count")
            episode_result = await session.run("MATCH (ep:Episode) RETURN count(ep) AS count")
            key_result = await session.run(
                "MATCH (k:IdentityKey) RETURN count(k) AS count"
            )
            memory_row = await memory_result.single()
            entity_row = await entity_result.single()
            relation_row = await relation_result.single()
            episode_row = await episode_result.single()
            key_row = await key_result.single()
        return GraphStoreStats(
            backend="neo4j",
            memory_count=int(memory_row["count"] if memory_row is not None else 0),
            entity_count=int(entity_row["count"] if entity_row is not None else 0),
            relation_count=int(relation_row["count"] if relation_row is not None else 0),
            episode_count=int(episode_row["count"] if episode_row is not None else 0),
            identity_key_count=int(key_row["count"] if key_row is not None else 0),
            cache_ready=self._search_cache is not None,
            cache_memory_count=len({episode.memory_id for episode in self._search_cache._episodes.values()})
            if self._search_cache is not None
            else 0,
            cache_entity_count=len(self._search_cache._entities) if self._search_cache is not None else 0,
            cache_relation_count=len(self._search_cache._relations) if self._search_cache is not None else 0,
            cache_episode_count=len(self._search_cache._episodes) if self._search_cache is not None else 0,
            cache_hydrated_at=self._cache_hydrated_at,
            cache_build_ms=self._cache_build_ms,
        )

    async def warm_cache(self) -> GraphStoreStats:
        await self._load_search_store()
        return await self.stats()

    async def resolve_identity(
        self, candidate: GraphIdentityCandidate
    ) -> IdentityResolutionResult:
        await self._ensure_ready()
        key = build_identity_key(candidate)
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (k:IdentityKey {key_id: $key_id})<-[:HAS_IDENTITY_KEY]-(e:Entity)
                RETURN collect(e.entity_id) AS entity_ids
                """,
                key_id=key.key_id,
            )
            row = await result.single()

        entity_ids = sorted(row["entity_ids"] or []) if row is not None else []
        if len(entity_ids) == 1:
            return IdentityResolutionResult(status="exact_match", entity_id=entity_ids[0], key=key)
        if len(entity_ids) > 1:
            return IdentityResolutionResult(
                status="candidate_match",
                key=key,
                candidates=[
                    IdentityResolutionCandidate(
                        entity_id=entity_id,
                        reason="identity_key_collision",
                        confidence=0.5,
                    )
                    for entity_id in entity_ids
                ],
            )
        return IdentityResolutionResult(status="no_match", key=key)

    async def passive_search(
        self,
        query_frame,
        *,
        max_entities: int = 5,
        max_relations: int = 8,
    ) -> GraphSearchResult:
        store = await self._load_search_store()
        return await store.passive_search(
            query_frame,
            max_entities=max_entities,
            max_relations=max_relations,
        )

    async def traverse(
        self,
        query_frame,
        *,
        seed_entity_ids: list[str] | None = None,
        max_hops: int = 2,
        max_entities: int = 10,
        max_relations: int = 12,
    ) -> GraphSearchResult:
        store = await self._load_search_store()
        return await store.traverse(
            query_frame,
            seed_entity_ids=seed_entity_ids,
            max_hops=max_hops,
            max_entities=max_entities,
            max_relations=max_relations,
        )

    async def get_entity(self, entity_id: str) -> GraphEntityNode | None:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                RETURN e
                """,
                entity_id=entity_id,
            )
            row = await result.single()
        if row is None:
            return None
        return _entity_from_props(dict(row["e"]))

    async def get_episode(self, episode_id: str) -> GraphEpisode | None:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (ep:Episode {episode_id: $episode_id})
                RETURN ep
                """,
                episode_id=episode_id,
            )
            row = await result.single()
        if row is None:
            return None
        return _episode_from_props(dict(row["ep"]))

    async def find_facts(self, query: GraphFactQuery) -> list[GraphRelationEdge]:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            result = await session.run(
                """
                MATCH (source:Entity)-[:OUTBOUND_RELATION]->(rel:Relation)-[:TARGETS]->(target:Entity)
                WHERE (
                    size($relation_types) = 0 OR rel.relation_type IN $relation_types
                )
                AND (
                    size($source_entity_ids) = 0 OR source.entity_id IN $source_entity_ids
                )
                AND (
                    size($target_entity_ids) = 0 OR target.entity_id IN $target_entity_ids
                )
                AND (
                    size($anchor_entity_ids) = 0
                    OR source.entity_id IN $anchor_entity_ids
                    OR target.entity_id IN $anchor_entity_ids
                )
                AND (
                    $active_only = false OR rel.invalidated_by_episode_id IS NULL
                )
                AND (
                    $valid_at_or_after IS NULL OR rel.invalid_at IS NULL OR rel.invalid_at >= $valid_at_or_after
                )
                AND (
                    $valid_at_or_before IS NULL OR rel.valid_at IS NULL OR rel.valid_at <= $valid_at_or_before
                )
                RETURN source.entity_id AS source_entity_id,
                       target.entity_id AS target_entity_id,
                       rel
                LIMIT $limit
                """,
                relation_types=[value.value for value in query.relation_types],
                source_entity_ids=query.source_entity_ids,
                target_entity_ids=query.target_entity_ids,
                anchor_entity_ids=query.anchor_entity_ids,
                active_only=query.active_only,
                valid_at_or_after=_serialize_datetime(query.valid_at_or_after),
                valid_at_or_before=_serialize_datetime(query.valid_at_or_before),
                limit=max(query.max_results * 3, query.max_results),
            )
            matches: list[GraphRelationEdge] = []
            async for row in result:
                relation = _relation_from_props(dict(row["rel"]))
                relation.source_entity_id = row["source_entity_id"]
                relation.target_entity_id = row["target_entity_id"]
                matches.append(relation)
        ranked = sorted(
            matches,
            key=lambda relation: _score_fact_query_match(relation, query),
            reverse=True,
        )
        return ranked[: query.max_results]

    async def close(self) -> None:
        if self.entity_index is not None:
            await self.entity_index.close()
        if self.adjudicator is not None:
            await self.adjudicator.close()
        if self.fact_adjudicator is not None:
            await self.fact_adjudicator.close()
        await self.driver.close()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        async with self.driver.session(database=self.database) as session:
            for statement in (
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT identity_key_unique IF NOT EXISTS FOR (k:IdentityKey) REQUIRE k.key_id IS UNIQUE",
                "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR (r:Relation) REQUIRE r.relation_id IS UNIQUE",
                "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (ep:Episode) REQUIRE ep.episode_id IS UNIQUE",
                "CREATE INDEX identity_key_type_idx IF NOT EXISTS FOR (k:IdentityKey) ON (k.key_type)",
                "CREATE INDEX relation_type_idx IF NOT EXISTS FOR (r:Relation) ON (r.relation_type)",
            ):
                result = await session.run(statement)
                await result.consume()
        self._ready = True

    async def _upsert_entity(self, session, *, document: GraphDocument, document_entity) -> tuple[GraphEntityNode, list[IdentityResolutionResult]]:
        memory_id = document.memory_id
        identity_candidates = list(document_entity.identity_candidates)
        for raw_value in [document_entity.canonical_name, *document_entity.alias_values]:
            identity_candidates.append(
                GraphIdentityCandidate(
                    key_type=IdentityKeyType.NAME_VARIANT,
                    raw_value=raw_value,
                    confidence=0.6,
                )
            )
        keys = [build_identity_key(candidate, memory_id=memory_id) for candidate in identity_candidates]

        key_lookup = await self._lookup_keys(session, keys)
        strong_matches: set[str] = set()
        weak_matches: set[str] = set()
        for key in keys:
            matched_entities = set(key_lookup.get(key.key_id, {}).get("entity_ids", []))
            if key.key_type in STRONG_KEY_TYPES:
                strong_matches.update(matched_entities)
            else:
                weak_matches.update(matched_entities)

        resolution_events: list[IdentityResolutionResult] = []
        if len(strong_matches) == 1:
            entity_id = next(iter(strong_matches))
            resolved_entity = await self.get_entity(entity_id)
            if resolved_entity is None:
                resolved_entity = _new_entity(memory_id, document_entity, provisional=False)
            resolution_events.append(
                IdentityResolutionResult(
                    status="exact_match",
                    entity_id=entity_id,
                    key=next((key for key in keys if key.key_type in STRONG_KEY_TYPES), None),
                )
            )
        elif len(strong_matches) > 1:
            resolved_entity = _new_entity(memory_id, document_entity, provisional=True)
            resolution_events.append(
                IdentityResolutionResult(
                    status="candidate_match",
                    entity_id=resolved_entity.entity_id,
                    candidates=[
                        IdentityResolutionCandidate(
                            entity_id=entity_id,
                            reason="multiple_strong_identity_matches",
                            confidence=0.5,
                        )
                        for entity_id in sorted(strong_matches)
                    ],
                )
            )
        elif entity_allows_name_auto_merge(document_entity.entity_type) and len(weak_matches) == 1:
            entity_id = next(iter(weak_matches))
            resolved_entity = await self.get_entity(entity_id)
            if resolved_entity is None:
                resolved_entity = _new_entity(memory_id, document_entity, provisional=False)
            resolution_events.append(
                IdentityResolutionResult(
                    status="exact_match",
                    entity_id=entity_id,
                )
            )
        else:
            similarity_hits = await self._similarity_hits(document_entity)
            decision = await self._resolve_ambiguous_entity(
                session=session,
                document=document,
                document_entity=document_entity,
                strong_match_ids=sorted(strong_matches),
                weak_match_ids=sorted(weak_matches),
                similarity_hits=similarity_hits,
            )
            if decision is None:
                resolved_entity = _new_entity(memory_id, document_entity, provisional=False)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="created_new",
                        entity_id=resolved_entity.entity_id,
                    )
                )
            elif decision.status == "exact_match" and decision.entity_id is not None:
                resolved_entity = await self.get_entity(decision.entity_id)
                if resolved_entity is None:
                    resolved_entity = _new_entity(memory_id, document_entity, provisional=False)
                    resolution_events.append(
                        IdentityResolutionResult(
                            status="created_new",
                            entity_id=resolved_entity.entity_id,
                        )
                    )
                else:
                    resolution_events.append(decision)
            elif decision.status == "candidate_match":
                resolved_entity = _new_entity(memory_id, document_entity, provisional=True)
                resolution_events.append(
                    decision.model_copy(update={"entity_id": resolved_entity.entity_id})
                )
            else:
                resolved_entity = _new_entity(memory_id, document_entity, provisional=False)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="created_new",
                        entity_id=resolved_entity.entity_id,
                    )
                )

        _merge_entity_model(
            memory_id=memory_id,
            entity=resolved_entity,
            document_entity=document_entity,
            keys=keys,
            existing_keys=key_lookup,
        )
        await self._persist_entity(session, resolved_entity)
        for key in keys:
            await self._persist_identity_key(session, key, resolved_entity.entity_id)

        return resolved_entity, resolution_events

    async def _upsert_relation(
        self,
        session,
        *,
        document: GraphDocument,
        episode: GraphEpisode,
        memory_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: RelationType,
        fact: str,
        confidence: float,
        valid_at,
        invalid_at,
        expires_at,
    ) -> tuple[GraphRelationEdge | None, list[str]]:
        relation_id = _deterministic_relation_id(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            fact=fact,
            valid_at=valid_at,
            invalid_at=invalid_at,
            expires_at=expires_at,
        )
        existing = await self._get_relation(session, relation_id)
        if existing is None:
            pending = PendingFactContext(
                relation_type=relation_type,
                source_entity_id=source_entity_id,
                source_entity_name=(await self.get_entity(source_entity_id)).canonical_name,
                target_entity_id=target_entity_id,
                target_entity_name=(await self.get_entity(target_entity_id)).canonical_name,
                fact=fact,
                confidence=confidence,
                valid_at=valid_at,
                invalid_at=invalid_at,
                expires_at=expires_at,
            )
            candidates = await self.find_facts(
                GraphFactQuery(
                    anchor_entity_ids=[source_entity_id, target_entity_id],
                    source_entity_ids=[source_entity_id],
                    relation_types=sorted(
                        compatible_relation_types(relation_type),
                        key=lambda value: value.value,
                    ),
                    active_only=True,
                    valid_at_or_after=valid_at,
                    valid_at_or_before=invalid_at,
                    max_results=8,
                )
            )
            exact_duplicate = next(
                (
                    candidate
                    for candidate in candidates
                    if exact_fact_signature(candidate) == exact_fact_signature(pending)
                ),
                None,
            )
            if exact_duplicate is not None:
                exact_duplicate.memory_ids = list(dict.fromkeys([*exact_duplicate.memory_ids, memory_id]))
                exact_duplicate.episode_ids = list(
                    dict.fromkeys([*exact_duplicate.episode_ids, episode.episode_id])
                )
                exact_duplicate.updated_at = utc_now()
                exact_duplicate.confidence = max(exact_duplicate.confidence, confidence)
                await self._persist_relation(session, exact_duplicate)
                await self._link_episode_to_relation(
                    session,
                    episode_id=episode.episode_id,
                    relation_id=exact_duplicate.relation_id,
                    edge_type="PRODUCED",
                )
                return exact_duplicate, []

            action, resolved_edge, invalidated_ids = await self._maybe_invalidate_relations(
                session=session,
                document=document,
                episode=episode,
                pending=pending,
                candidates=candidates,
            )
            if action == "discard":
                return resolved_edge, invalidated_ids
            if action == "merge" and resolved_edge is not None:
                await self._link_episode_to_relation(
                    session,
                    episode_id=episode.episode_id,
                    relation_id=resolved_edge.relation_id,
                    edge_type="PRODUCED",
                )
                return resolved_edge, invalidated_ids
            relation = GraphRelationEdge(
                relation_id=relation_id,
                relation_type=relation_type,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                fact=fact,
                memory_ids=[memory_id],
                episode_ids=[episode.episode_id],
                confidence=confidence,
                valid_at=valid_at,
                invalid_at=invalid_at,
                expires_at=expires_at,
            )
            await self._persist_relation(session, relation)
            await self._link_episode_to_relation(
                session,
                episode_id=episode.episode_id,
                relation_id=relation.relation_id,
                edge_type="PRODUCED",
            )
            return relation, invalidated_ids

        relation = existing
        relation.memory_ids = list(dict.fromkeys([*relation.memory_ids, memory_id]))
        relation.episode_ids = list(dict.fromkeys([*relation.episode_ids, episode.episode_id]))
        relation.updated_at = utc_now()
        relation.confidence = max(relation.confidence, confidence)
        relation.valid_at = relation.valid_at or valid_at
        relation.invalid_at = invalid_at or relation.invalid_at
        relation.expires_at = expires_at or relation.expires_at
        await self._persist_relation(session, relation)
        await self._link_episode_to_relation(
            session,
            episode_id=episode.episode_id,
            relation_id=relation.relation_id,
            edge_type="PRODUCED",
        )
        return relation, []

    async def _persist_entity(self, session, entity: GraphEntityNode) -> None:
        await session.run(
            """
            MERGE (e:Entity {entity_id: $entity_id})
            SET e.entity_type = $entity_type,
                e.canonical_name = $canonical_name,
                e.alias_values = $alias_values,
                e.identity_key_ids = $identity_key_ids,
                e.attributes_json = $attributes_json,
                e.memory_ids = $memory_ids,
                e.created_at = coalesce(e.created_at, $created_at),
                e.updated_at = $updated_at,
                e.resolution_state = $resolution_state
            """,
            entity_id=entity.entity_id,
            entity_type=entity.entity_type.value,
            canonical_name=entity.canonical_name,
            alias_values=entity.alias_values,
            identity_key_ids=entity.identity_key_ids,
            attributes_json=json.dumps(entity.attributes, sort_keys=True),
            memory_ids=entity.memory_ids,
            created_at=_serialize_datetime(entity.created_at),
            updated_at=_serialize_datetime(entity.updated_at),
            resolution_state=entity.resolution_state,
        )

    async def _persist_identity_key(
        self,
        session,
        key: GraphIdentityKey,
        entity_id: str,
    ) -> None:
        await session.run(
            """
            MERGE (k:IdentityKey {key_id: $key_id})
            SET k.key_type = $key_type,
                k.normalized_value = $normalized_value,
                k.raw_values = $raw_values,
                k.provider = $provider,
                k.confidence = $confidence,
                k.first_seen_at = coalesce(k.first_seen_at, $first_seen_at),
                k.last_seen_at = $last_seen_at,
                k.memory_ids = $memory_ids
            WITH k
            MATCH (e:Entity {entity_id: $entity_id})
            MERGE (e)-[:HAS_IDENTITY_KEY]->(k)
            """,
            key_id=key.key_id,
            key_type=key.key_type.value,
            normalized_value=key.normalized_value,
            raw_values=key.raw_values,
            provider=key.provider,
            confidence=key.confidence,
            first_seen_at=_serialize_datetime(key.first_seen_at),
            last_seen_at=_serialize_datetime(key.last_seen_at),
            memory_ids=key.memory_ids,
            entity_id=entity_id,
        )

    async def _persist_episode(self, session, episode: GraphEpisode) -> None:
        await session.run(
            """
            MERGE (ep:Episode {episode_id: $episode_id})
            SET ep.memory_id = $memory_id,
                ep.source_type = $source_type,
                ep.provenance_source_kind = $provenance_source_kind,
                ep.provenance_source_id = $provenance_source_id,
                ep.session_id = $session_id,
                ep.task_id = $task_id,
                ep.channel = $channel,
                ep.created_at = $created_at,
                ep.extracted_at = $extracted_at,
                ep.extraction_confidence = $extraction_confidence,
                ep.rationale = $rationale,
                ep.source_excerpt = $source_excerpt,
                ep.produced_relation_ids = $produced_relation_ids,
                ep.invalidated_relation_ids = $invalidated_relation_ids
            """,
            episode_id=episode.episode_id,
            memory_id=episode.memory_id,
            source_type=episode.source_type,
            provenance_source_kind=episode.provenance_source_kind,
            provenance_source_id=episode.provenance_source_id,
            session_id=episode.session_id,
            task_id=episode.task_id,
            channel=episode.channel,
            created_at=_serialize_datetime(episode.created_at),
            extracted_at=_serialize_datetime(episode.extracted_at),
            extraction_confidence=episode.extraction_confidence,
            rationale=episode.rationale,
            source_excerpt=episode.source_excerpt,
            produced_relation_ids=episode.produced_relation_ids,
            invalidated_relation_ids=episode.invalidated_relation_ids,
        )

    async def _persist_relation(self, session, relation: GraphRelationEdge) -> None:
        await session.run(
            """
            MERGE (rel:Relation {relation_id: $relation_id})
            SET rel.relation_type = $relation_type,
                rel.fact = $fact,
                rel.memory_ids = $memory_ids,
                rel.episode_ids = $episode_ids,
                rel.confidence = $confidence,
                rel.created_at = coalesce(rel.created_at, $created_at),
                rel.updated_at = $updated_at,
                rel.valid_at = $valid_at,
                rel.invalid_at = $invalid_at,
                rel.expires_at = $expires_at,
                rel.invalidated_by_episode_id = $invalidated_by_episode_id,
                rel.source_entity_id = $source_entity_id,
                rel.target_entity_id = $target_entity_id
            WITH rel
            MATCH (source:Entity {entity_id: $source_entity_id})
            MATCH (target:Entity {entity_id: $target_entity_id})
            MERGE (source)-[:OUTBOUND_RELATION]->(rel)
            MERGE (rel)-[:TARGETS]->(target)
            """,
            relation_id=relation.relation_id,
            relation_type=relation.relation_type.value,
            fact=relation.fact,
            memory_ids=relation.memory_ids,
            episode_ids=relation.episode_ids,
            confidence=relation.confidence,
            created_at=_serialize_datetime(relation.created_at),
            updated_at=_serialize_datetime(relation.updated_at),
            valid_at=_serialize_datetime(relation.valid_at),
            invalid_at=_serialize_datetime(relation.invalid_at),
            expires_at=_serialize_datetime(relation.expires_at),
            invalidated_by_episode_id=relation.invalidated_by_episode_id,
            source_entity_id=relation.source_entity_id,
            target_entity_id=relation.target_entity_id,
        )

    async def _link_episode_to_relation(
        self,
        session,
        *,
        episode_id: str,
        relation_id: str,
        edge_type: str,
    ) -> None:
        if edge_type not in {"PRODUCED", "INVALIDATED"}:
            raise ValueError(f"Unsupported episode edge type: {edge_type}")
        await session.run(
            f"""
            MATCH (ep:Episode {{episode_id: $episode_id}})
            MATCH (rel:Relation {{relation_id: $relation_id}})
            MERGE (ep)-[:{edge_type}]->(rel)
            """,
            episode_id=episode_id,
            relation_id=relation_id,
        )

    async def _lookup_keys(self, session, keys: list[GraphIdentityKey]) -> dict[str, dict[str, object]]:
        if not keys:
            return {}
        key_by_id = {key.key_id: key for key in keys}
        result = await session.run(
            """
            UNWIND $key_ids AS key_id
            OPTIONAL MATCH (k:IdentityKey {key_id: key_id})<-[:HAS_IDENTITY_KEY]-(e:Entity)
            RETURN key_id, head(collect(k)) AS key_node, collect(e.entity_id) AS entity_ids
            """,
            key_ids=list(key_by_id.keys()),
        )
        lookup: dict[str, dict[str, object]] = {}
        async for row in result:
            node = row["key_node"]
            lookup[row["key_id"]] = {
                "entity_ids": sorted(set(row["entity_ids"] or [])),
                "key": _identity_key_from_props(dict(node)) if node is not None else None,
            }
        return lookup

    async def _get_relation(self, session, relation_id: str) -> GraphRelationEdge | None:
        result = await session.run(
            """
            MATCH (rel:Relation {relation_id: $relation_id})
            RETURN rel
            """,
            relation_id=relation_id,
        )
        row = await result.single()
        if row is None:
            return None
        return _relation_from_props(dict(row["rel"]))

    async def _maybe_invalidate_relations(
        self,
        *,
        session,
        document: GraphDocument,
        episode: GraphEpisode,
        pending: PendingFactContext,
        candidates: list[GraphRelationEdge],
    ) -> tuple[str, GraphRelationEdge | None, list[str]]:
        if not candidates:
            return "create", None, []

        deterministic_invalidated_ids = _deterministic_invalidation_candidate_ids(
            pending=pending,
            candidates=candidates,
            pending_effective_valid_at=pending.valid_at or episode.created_at,
        )
        if deterministic_invalidated_ids:
            for relation_id in deterministic_invalidated_ids:
                relation = await self._get_relation(session, relation_id)
                if relation is None or not _is_active_relation(relation):
                    continue
                relation.invalidated_by_episode_id = episode.episode_id
                relation.updated_at = utc_now()
                await self._persist_relation(session, relation)
                await self._link_episode_to_relation(
                    session,
                    episode_id=episode.episode_id,
                    relation_id=relation.relation_id,
                    edge_type="INVALIDATED",
                )
            return "create", None, deterministic_invalidated_ids

        candidate_contexts = []
        for candidate in candidates:
            source = await self.get_entity(candidate.source_entity_id)
            target = await self.get_entity(candidate.target_entity_id)
            candidate_contexts.append(
                FactCandidateContext(
                    relation_id=candidate.relation_id,
                    relation_type=candidate.relation_type,
                    source_entity_id=candidate.source_entity_id,
                    source_entity_name=source.canonical_name if source is not None else candidate.source_entity_id,
                    target_entity_id=candidate.target_entity_id,
                    target_entity_name=target.canonical_name if target is not None else candidate.target_entity_id,
                    fact=candidate.fact,
                    confidence=candidate.confidence,
                    memory_count=len(candidate.memory_ids),
                    episode_count=len(candidate.episode_ids),
                    valid_at=candidate.valid_at,
                    invalid_at=candidate.invalid_at,
                    expires_at=candidate.expires_at,
                    active=_is_active_relation(candidate),
                    retrieval_reason="active_relation_family_candidate",
                )
            )
        if self.fact_adjudicator is None:
            return "create", None, []

        timezone_name = getattr(self.fact_adjudicator, "timezone_name", "UTC")
        try:
            decision = await self.fact_adjudicator.adjudicate(
                FactAdjudicationRequest(
                    memory_id=document.memory_id,
                    episode=episode,
                    pending_fact=pending,
                    candidate_facts=candidate_contexts,
                    source_text=document.source_text,
                    utc_time_anchor=utc_now(),
                    local_time_anchor=build_local_time_anchor(timezone_name).isoformat(),
                    provenance_created_at=episode.created_at,
                    timezone_name=timezone_name,
                )
            )
        except Exception:
            logger.exception("Fact adjudication failed for memory %s", document.memory_id)
            return "create", None, []

        if decision.decision == "discard_new":
            return "discard", None, []
        if decision.decision == "merge_with_existing" and decision.chosen_relation_id:
            existing = await self._get_relation(session, decision.chosen_relation_id)
            if existing is not None:
                existing.memory_ids = list(dict.fromkeys([*existing.memory_ids, document.memory_id]))
                existing.episode_ids = list(dict.fromkeys([*existing.episode_ids, episode.episode_id]))
                existing.updated_at = utc_now()
                await self._persist_relation(session, existing)
                return "merge", existing, []
        if decision.decision != "invalidate_existing":
            return "create", None, []

        invalidated_ids: list[str] = []
        for relation_id in decision.invalidated_relation_ids:
            relation = await self._get_relation(session, relation_id)
            if relation is None or not _is_active_relation(relation):
                continue
            relation.invalidated_by_episode_id = episode.episode_id
            relation.updated_at = utc_now()
            await self._persist_relation(session, relation)
            await self._link_episode_to_relation(
                session,
                episode_id=episode.episode_id,
                relation_id=relation.relation_id,
                edge_type="INVALIDATED",
            )
            invalidated_ids.append(relation.relation_id)
        return "create", None, invalidated_ids

    async def _load_search_store(self) -> InMemoryGraphStore:
        await self._ensure_ready()
        if self._search_cache is not None:
            return self._search_cache
        async with self._cache_lock:
            if self._search_cache is not None:
                return self._search_cache
            store = await self._hydrate_search_store()
            self._search_cache = store
            return store

    async def _hydrate_search_store(self) -> InMemoryGraphStore:
        hydration_started = perf_counter()
        store = InMemoryGraphStore(entity_index=self.entity_index)
        async with self.driver.session(database=self.database) as session:
            entity_result = await session.run(
                """
                MATCH (e:Entity)
                RETURN e
                """
            )
            entities: dict[str, GraphEntityNode] = {}
            async for row in entity_result:
                entity = _entity_from_props(dict(row["e"]))
                entities[entity.entity_id] = entity

            key_result = await session.run(
                """
                MATCH (k:IdentityKey)
                RETURN k
                """
            )
            identity_keys: dict[str, GraphIdentityKey] = {}
            async for row in key_result:
                key = _identity_key_from_props(dict(row["k"]))
                identity_keys[key.key_id] = key

            mapping_result = await session.run(
                """
                MATCH (e:Entity)-[:HAS_IDENTITY_KEY]->(k:IdentityKey)
                RETURN e.entity_id AS entity_id, k.key_id AS key_id
                """
            )
            key_to_entities: dict[str, set[str]] = {}
            async for row in mapping_result:
                key_id = row["key_id"]
                entity_id = row["entity_id"]
                key_to_entities.setdefault(key_id, set()).add(entity_id)
                entity = entities.get(entity_id)
                if entity is not None and key_id not in entity.identity_key_ids:
                    entity.identity_key_ids.append(key_id)

            relation_result = await session.run(
                """
                MATCH (source:Entity)-[:OUTBOUND_RELATION]->(rel:Relation)-[:TARGETS]->(target:Entity)
                RETURN source.entity_id AS source_entity_id,
                       target.entity_id AS target_entity_id,
                       rel
                """
            )
            relations: dict[str, GraphRelationEdge] = {}
            entity_to_relations: dict[str, set[str]] = {}
            async for row in relation_result:
                relation = _relation_from_props(dict(row["rel"]))
                relation.source_entity_id = row["source_entity_id"]
                relation.target_entity_id = row["target_entity_id"]
                relations[relation.relation_id] = relation
                entity_to_relations.setdefault(relation.source_entity_id, set()).add(
                    relation.relation_id
                )
                entity_to_relations.setdefault(relation.target_entity_id, set()).add(
                    relation.relation_id
                )

            episode_result = await session.run(
                """
                MATCH (ep:Episode)
                RETURN ep
                """
            )
            episodes: dict[str, GraphEpisode] = {}
            async for row in episode_result:
                episode = _episode_from_props(dict(row["ep"]))
                episodes[episode.episode_id] = episode

        store._entities = entities
        store._episodes = episodes
        store._identity_keys = identity_keys
        store._relations = relations
        store._key_to_entities = defaultdict(set, key_to_entities)
        store._entity_to_relations = defaultdict(set, entity_to_relations)
        self._cache_hydrated_at = utc_now()
        self._cache_build_ms = round((perf_counter() - hydration_started) * 1000.0, 3)
        logger.info(
            "memory.graph_cache_hydrated backend=neo4j entities=%s relations=%s episodes=%s memory_count=%s build_ms=%s",
            len(entities),
            len(relations),
            len(episodes),
            len({episode.memory_id for episode in episodes.values()}),
            self._cache_build_ms,
        )
        return store

    async def _cache_ingest_document(self, document: GraphDocument) -> None:
        async with self._cache_lock:
            if self.adjudicator is not None or self.fact_adjudicator is not None:
                self._search_cache = None
                self._cache_hydrated_at = None
                self._cache_build_ms = None
                return
            if self._search_cache is None:
                self._search_cache = await self._hydrate_search_store()
                return
            await self._search_cache.ingest_document(document)

    async def _cache_remove_memory(self, memory_id: str) -> None:
        async with self._cache_lock:
            if self.adjudicator is not None or self.fact_adjudicator is not None:
                self._search_cache = None
                self._cache_hydrated_at = None
                self._cache_build_ms = None
                return
            if self._search_cache is None:
                return
            await self._search_cache.remove_memory(memory_id)

    async def _similarity_hits(self, document_entity) -> list:
        if self.entity_index is None:
            return []
        return await self.entity_index.search(
            entity_similarity_text_parts(
                entity_type=document_entity.entity_type,
                canonical_name=document_entity.canonical_name,
                alias_values=document_entity.alias_values,
                attributes=document_entity.attributes,
            ),
            entity_types=sorted(
                similarity_candidate_types(document_entity.entity_type),
                key=lambda value: value.value,
            ),
            limit=5,
        )

    async def _resolve_ambiguous_entity(
        self,
        *,
        session,
        document: GraphDocument,
        document_entity,
        strong_match_ids: list[str],
        weak_match_ids: list[str],
        similarity_hits: list,
    ) -> IdentityResolutionResult | None:
        candidate_contexts = await self._build_candidate_contexts(
            session,
            strong_match_ids=strong_match_ids,
            weak_match_ids=weak_match_ids,
            similarity_hits=similarity_hits,
        )
        if self.adjudicator is not None and candidate_contexts:
            timezone_name = getattr(self.adjudicator, "timezone_name", "UTC")
            try:
                decision = await self.adjudicator.adjudicate(
                    EntityAdjudicationRequest(
                        memory_id=document.memory_id,
                        pending_entity=document_entity,
                        candidate_entities=candidate_contexts,
                        source_text=document.source_text,
                        utc_time_anchor=utc_now(),
                        local_time_anchor=build_local_time_anchor(timezone_name),
                        provenance_created_at=document.created_at,
                        timezone_name=timezone_name,
                    )
                )
            except Exception:
                logger.exception("Entity adjudication failed for memory %s", document.memory_id)
            else:
                coerced = self._decision_to_resolution_result(decision, candidate_contexts)
                if coerced is not None:
                    return coerced

        if strong_match_ids:
            return IdentityResolutionResult(
                status="candidate_match",
                candidates=[
                    IdentityResolutionCandidate(
                        entity_id=entity_id,
                        reason="multiple_strong_identity_matches",
                        confidence=0.5,
                    )
                    for entity_id in strong_match_ids
                ],
            )
        if weak_match_ids:
            return IdentityResolutionResult(
                status="candidate_match",
                candidates=[
                    IdentityResolutionCandidate(
                        entity_id=entity_id,
                        reason="weak_alias_only_match",
                        confidence=0.35,
                    )
                    for entity_id in weak_match_ids
                ],
            )
        auto_merge_hits = [hit for hit in similarity_hits if hit.score >= 0.92]
        candidate_hits = [hit for hit in similarity_hits if 0.78 <= hit.score < 0.92]
        if auto_merge_hits:
            return IdentityResolutionResult(
                status="exact_match",
                entity_id=auto_merge_hits[0].entity_id,
            )
        if candidate_hits:
            return IdentityResolutionResult(
                status="candidate_match",
                candidates=[
                    IdentityResolutionCandidate(
                        entity_id=hit.entity_id,
                        reason="entity_similarity_candidate",
                        confidence=min(max(hit.score, 0.0), 1.0),
                    )
                    for hit in candidate_hits
                ],
            )
        return None

    async def _build_candidate_contexts(
        self,
        session,
        *,
        strong_match_ids: list[str],
        weak_match_ids: list[str],
        similarity_hits: list,
    ) -> list[EntityCandidateContext]:
        candidates: dict[str, EntityCandidateContext] = {}

        for entity_id in strong_match_ids:
            context = await self._candidate_context_for_entity_id(
                session,
                entity_id,
                match_reason="multiple_strong_identity_matches",
            )
            if context is not None:
                candidates[entity_id] = context

        for entity_id in weak_match_ids:
            context = await self._candidate_context_for_entity_id(
                session,
                entity_id,
                match_reason="weak_alias_only_match",
            )
            if context is None:
                continue
            existing = candidates.get(entity_id)
            if existing is None:
                candidates[entity_id] = context
            else:
                existing.match_reasons = list(
                    dict.fromkeys([*existing.match_reasons, *context.match_reasons])
                )

        for hit in similarity_hits:
            context = await self._candidate_context_for_entity_id(
                session,
                hit.entity_id,
                match_reason="entity_similarity_candidate",
                similarity_score=min(max(hit.score, 0.0), 1.0),
            )
            if context is None:
                continue
            existing = candidates.get(hit.entity_id)
            if existing is None:
                candidates[hit.entity_id] = context
            else:
                existing.match_reasons = list(
                    dict.fromkeys([*existing.match_reasons, *context.match_reasons])
                )
                existing.similarity_score = max(
                    existing.similarity_score or 0.0,
                    context.similarity_score or 0.0,
                )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                max(item.similarity_score or 0.0, 0.0),
                len(item.match_reasons),
                item.memory_count,
            ),
            reverse=True,
        )
        return ranked[:5]

    async def _candidate_context_for_entity_id(
        self,
        session,
        entity_id: str,
        *,
        match_reason: str,
        similarity_score: float | None = None,
    ) -> EntityCandidateContext | None:
        entity = await self.get_entity(entity_id)
        if entity is None:
            return None
        result = await session.run(
            """
            MATCH (e:Entity {entity_id: $entity_id})
            OPTIONAL MATCH (e)-[:OUTBOUND_RELATION]->(rel:Relation)-[:TARGETS]->(target:Entity)
            RETURN rel.relation_type AS relation_type,
                   target.canonical_name AS target_name
            LIMIT 4
            """,
            entity_id=entity_id,
        )
        relation_summaries: list[str] = []
        async for row in result:
            relation_type = row["relation_type"]
            target_name = row["target_name"]
            if relation_type and target_name:
                relation_summaries.append(f"{relation_type} -> {target_name}")
        return EntityCandidateContext(
            entity_id=entity.entity_id,
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            alias_values=entity.alias_values,
            attributes=entity.attributes,
            resolution_state=entity.resolution_state,
            memory_count=len(entity.memory_ids),
            relation_summaries=relation_summaries,
            similarity_score=similarity_score,
            match_reasons=[match_reason],
        )

    @staticmethod
    def _decision_to_resolution_result(
        decision,
        candidate_contexts: list[EntityCandidateContext],
    ) -> IdentityResolutionResult | None:
        candidate_map = {candidate.entity_id: candidate for candidate in candidate_contexts}
        if decision.decision == "exact_match":
            if decision.chosen_entity_id not in candidate_map:
                return None
            return IdentityResolutionResult(
                status="exact_match",
                entity_id=decision.chosen_entity_id,
            )
        if decision.decision == "candidate_match":
            candidate_ids = [
                entity_id
                for entity_id in decision.candidate_entity_ids
                if entity_id in candidate_map
            ] or [candidate.entity_id for candidate in candidate_contexts]
            return IdentityResolutionResult(
                status="candidate_match",
                candidates=[
                    IdentityResolutionCandidate(
                        entity_id=entity_id,
                        reason="llm_entity_adjudication_candidate",
                        confidence=min(max(decision.confidence, 0.0), 1.0),
                    )
                    for entity_id in candidate_ids
                ],
            )
        if decision.decision == "created_new":
            return IdentityResolutionResult(status="created_new")
        return None


def _new_entity(memory_id: str, document_entity, *, provisional: bool) -> GraphEntityNode:
    return GraphEntityNode(
        entity_type=document_entity.entity_type,
        canonical_name=document_entity.canonical_name,
        alias_values=list(dict.fromkeys(document_entity.alias_values)),
        attributes=dict(document_entity.attributes),
        memory_ids=[memory_id],
        resolution_state="provisional" if provisional else "canonical",
    )


def _merge_entity_model(
    *,
    memory_id: str,
    entity: GraphEntityNode,
    document_entity,
    keys: list[GraphIdentityKey],
    existing_keys: dict[str, dict[str, object]],
) -> None:
    entity.updated_at = utc_now()
    entity.canonical_name = entity.canonical_name or document_entity.canonical_name
    entity.alias_values = list(
        dict.fromkeys(
            [*entity.alias_values, document_entity.canonical_name, *document_entity.alias_values]
        )
    )
    entity.memory_ids = list(dict.fromkeys([*entity.memory_ids, memory_id]))
    entity.attributes.update(document_entity.attributes)

    if entity.resolution_state == "provisional" and any(
        key.key_type in STRONG_KEY_TYPES for key in keys
    ):
        entity.resolution_state = "canonical"

    for key in keys:
        existing = existing_keys.get(key.key_id, {}).get("key")
        if isinstance(existing, GraphIdentityKey):
            key.raw_values = list(dict.fromkeys([*existing.raw_values, *key.raw_values]))
            key.first_seen_at = existing.first_seen_at
            key.last_seen_at = utc_now()
            key.memory_ids = list(dict.fromkeys([*existing.memory_ids, memory_id]))
            key.confidence = max(existing.confidence, key.confidence)
        if key.key_id not in entity.identity_key_ids:
            entity.identity_key_ids.append(key.key_id)


def _entity_from_props(props: dict) -> GraphEntityNode:
    return GraphEntityNode(
        entity_id=props["entity_id"],
        entity_type=props["entity_type"],
        canonical_name=props["canonical_name"],
        alias_values=list(props.get("alias_values") or []),
        identity_key_ids=list(props.get("identity_key_ids") or []),
        attributes=_deserialize_json(props.get("attributes_json")),
        memory_ids=list(props.get("memory_ids") or []),
        created_at=_deserialize_datetime(props.get("created_at")) or utc_now(),
        updated_at=_deserialize_datetime(props.get("updated_at")) or utc_now(),
        resolution_state=props.get("resolution_state") or "canonical",
    )


def _identity_key_from_props(props: dict) -> GraphIdentityKey:
    return GraphIdentityKey(
        key_id=props["key_id"],
        key_type=props["key_type"],
        normalized_value=props["normalized_value"],
        raw_values=list(props.get("raw_values") or []),
        provider=props.get("provider"),
        confidence=float(props.get("confidence") or 1.0),
        first_seen_at=_deserialize_datetime(props.get("first_seen_at")) or utc_now(),
        last_seen_at=_deserialize_datetime(props.get("last_seen_at")) or utc_now(),
        memory_ids=list(props.get("memory_ids") or []),
    )


def _relation_from_props(props: dict) -> GraphRelationEdge:
    return GraphRelationEdge(
        relation_id=props["relation_id"],
        relation_type=props["relation_type"],
        source_entity_id=props.get("source_entity_id", ""),
        target_entity_id=props.get("target_entity_id", ""),
        fact=props["fact"],
        memory_ids=list(props.get("memory_ids") or []),
        episode_ids=list(props.get("episode_ids") or []),
        confidence=float(props.get("confidence") or 1.0),
        created_at=_deserialize_datetime(props.get("created_at")) or utc_now(),
        updated_at=_deserialize_datetime(props.get("updated_at")) or utc_now(),
        valid_at=_deserialize_datetime(props.get("valid_at")),
        invalid_at=_deserialize_datetime(props.get("invalid_at")),
        expires_at=_deserialize_datetime(props.get("expires_at")),
        invalidated_by_episode_id=props.get("invalidated_by_episode_id"),
    )


def _episode_from_props(props: dict) -> GraphEpisode:
    return GraphEpisode(
        episode_id=props["episode_id"],
        memory_id=props["memory_id"],
        source_type=props["source_type"],
        provenance_source_kind=props.get("provenance_source_kind"),
        provenance_source_id=props.get("provenance_source_id"),
        session_id=props.get("session_id"),
        task_id=props.get("task_id"),
        channel=props.get("channel"),
        created_at=_deserialize_datetime(props.get("created_at")) or utc_now(),
        extracted_at=_deserialize_datetime(props.get("extracted_at")) or utc_now(),
        extraction_confidence=float(props.get("extraction_confidence") or 1.0),
        rationale=props.get("rationale"),
        source_excerpt=props.get("source_excerpt"),
        produced_relation_ids=list(props.get("produced_relation_ids") or []),
        invalidated_relation_ids=list(props.get("invalidated_relation_ids") or []),
    )


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _deserialize_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "isoformat"):
        try:
            return datetime.fromisoformat(value.isoformat())
        except ValueError:
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _deserialize_json(value) -> dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _deterministic_relation_id(
    *,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: RelationType,
    fact: str,
    valid_at,
    invalid_at,
    expires_at,
) -> str:
    payload = "||".join(
        [
            source_entity_id,
            target_entity_id,
            relation_type.value,
            fact,
            valid_at.isoformat() if valid_at is not None else "",
            invalid_at.isoformat() if invalid_at is not None else "",
            expires_at.isoformat() if expires_at is not None else "",
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"rel_{digest[:24]}"


def _coerce_episode(document: GraphDocument) -> GraphEpisode:
    episode = document.episode
    if episode is None:
        episode = GraphEpisode(
            memory_id=document.memory_id,
            source_type="unknown",
            created_at=document.created_at,
        )
    if not episode.source_excerpt:
        episode.source_excerpt = _excerpt(document.source_text)
    return episode


def _excerpt(value: str | None, *, limit: int = 280) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _deterministic_invalidation_candidate_ids(
    *,
    pending: PendingFactContext,
    candidates: list[GraphRelationEdge],
    pending_effective_valid_at,
) -> list[str]:
    if pending_effective_valid_at is None:
        return []

    invalidated_ids: list[str] = []
    pending_fact_key = " ".join(pending.fact.casefold().split())
    for candidate in candidates:
        if candidate.invalidated_by_episode_id is not None:
            continue
        if candidate.relation_type != pending.relation_type:
            continue
        if candidate.source_entity_id != pending.source_entity_id:
            continue
        if candidate.target_entity_id != pending.target_entity_id:
            continue

        candidate_effective_valid_at = candidate.valid_at or candidate.updated_at or candidate.created_at
        if candidate_effective_valid_at is None:
            continue
        if pending_effective_valid_at <= candidate_effective_valid_at:
            continue

        candidate_fact_key = " ".join(candidate.fact.casefold().split())
        if candidate_fact_key == pending_fact_key:
            continue

        invalidated_ids.append(candidate.relation_id)
    return invalidated_ids


def _is_active_relation(relation: GraphRelationEdge) -> bool:
    now = utc_now()
    if relation.invalidated_by_episode_id is not None:
        return False
    if relation.invalid_at and relation.invalid_at <= now:
        return False
    if relation.expires_at and relation.expires_at <= now:
        return False
    return True


def _score_fact_query_match(relation: GraphRelationEdge, query: GraphFactQuery) -> float:
    score = 0.0
    anchor_ids = set(query.anchor_entity_ids)
    if relation.source_entity_id in anchor_ids:
        score += 0.6
    if relation.target_entity_id in anchor_ids:
        score += 0.45
    if query.source_entity_ids and relation.source_entity_id in set(query.source_entity_ids):
        score += 0.8
    if query.target_entity_ids and relation.target_entity_id in set(query.target_entity_ids):
        score += 0.6
    if query.relation_types and relation.relation_type in set(query.relation_types):
        score += 0.35
    if _is_active_relation(relation):
        score += 0.2
    score += min(max(relation.confidence, 0.0), 1.0) * 0.05
    score += len(relation.episode_ids) * 0.01
    return score
