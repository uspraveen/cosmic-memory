"""In-memory graph store for local development and contract testing."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict, deque
from datetime import timezone

from cosmic_memory.domain.models import GraphStoreStats, utc_now
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
    relation_fact_signature_key,
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
    GraphQueryFrame,
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


class InMemoryGraphStore:
    """Small in-memory graph store with strict identity merge rules."""

    def __init__(
        self,
        *,
        entity_index: EntitySimilarityIndex | None = None,
        adjudicator: EntityAdjudicationService | None = None,
        fact_adjudicator: FactAdjudicationService | None = None,
    ) -> None:
        self._entities: dict[str, GraphEntityNode] = {}
        self._episodes: dict[str, GraphEpisode] = {}
        self._identity_keys: dict[str, GraphIdentityKey] = {}
        self._entity_to_relations: dict[str, set[str]] = defaultdict(set)
        self._relations: dict[str, GraphRelationEdge] = {}
        self._key_to_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_index = entity_index
        self.adjudicator = adjudicator
        self.fact_adjudicator = fact_adjudicator

    async def ingest_document(self, document: GraphDocument) -> GraphIngestResult:
        ref_to_entity_id: dict[str, str] = {}
        resolution_events: list[IdentityResolutionResult] = []
        changed_entity_ids: set[str] = set()
        episode = self._upsert_episode(document)

        for entity in document.entities:
            resolved_entity, events = await self._upsert_entity(document, entity)
            ref_to_entity_id[entity.local_ref] = resolved_entity.entity_id
            changed_entity_ids.add(resolved_entity.entity_id)
            resolution_events.extend(events)

        relation_ids: list[str] = []
        invalidated_relation_ids: list[str] = []
        for relation in document.relations:
            source_entity_id = ref_to_entity_id.get(relation.source_ref)
            target_entity_id = ref_to_entity_id.get(relation.target_ref)
            if source_entity_id is None or target_entity_id is None:
                continue
            edge, invalidated_ids = await self._upsert_relation(
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
        self._episodes[episode.episode_id] = episode

        if self.entity_index is not None and changed_entity_ids:
            await self.entity_index.sync_entities(
                [self._entities[entity_id] for entity_id in changed_entity_ids if entity_id in self._entities]
            )

        return GraphIngestResult(
            memory_id=document.memory_id,
            episode_id=episode.episode_id,
            entity_ids=list(dict.fromkeys(ref_to_entity_id.values())),
            relation_ids=list(dict.fromkeys(relation_ids)),
            invalidated_relation_ids=list(dict.fromkeys(invalidated_relation_ids)),
            resolution_events=resolution_events,
        )

    async def resolve_identity(
        self, candidate: GraphIdentityCandidate
    ) -> IdentityResolutionResult:
        key = build_identity_key(candidate)
        entity_ids = sorted(self._key_to_entities.get(key.key_id, set()))
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

    async def remove_memory(self, memory_id: str) -> None:
        episode_ids_to_remove = [
            episode_id
            for episode_id, episode in self._episodes.items()
            if episode.memory_id == memory_id
        ]
        relation_ids_to_delete: list[str] = []
        touched_entity_ids: set[str] = set()
        for relation_id, relation in self._relations.items():
            if memory_id not in relation.memory_ids:
                if relation.invalidated_by_episode_id in episode_ids_to_remove:
                    relation.invalidated_by_episode_id = None
                    relation.updated_at = utc_now()
                continue
            relation.memory_ids = [value for value in relation.memory_ids if value != memory_id]
            relation.episode_ids = [
                value for value in relation.episode_ids if value not in episode_ids_to_remove
            ]
            if relation.invalidated_by_episode_id in episode_ids_to_remove:
                relation.invalidated_by_episode_id = None
            if relation.memory_ids:
                relation.updated_at = utc_now()
                continue
            relation_ids_to_delete.append(relation_id)

        for relation_id in relation_ids_to_delete:
            relation = self._relations.pop(relation_id)
            self._entity_to_relations[relation.source_entity_id].discard(relation_id)
            self._entity_to_relations[relation.target_entity_id].discard(relation_id)

        for entity in self._entities.values():
            if memory_id in entity.memory_ids:
                entity.memory_ids = [value for value in entity.memory_ids if value != memory_id]
                touched_entity_ids.add(entity.entity_id)

        for key in self._identity_keys.values():
            if memory_id in key.memory_ids:
                key.memory_ids = [value for value in key.memory_ids if value != memory_id]

        for episode_id in episode_ids_to_remove:
            self._episodes.pop(episode_id, None)

        if self.entity_index is not None:
            await self.entity_index.delete_entities(
                [entity_id for entity_id in touched_entity_ids if entity_id in self._entities and not self._entities[entity_id].memory_ids]
            )
            await self.entity_index.sync_entities(
                [self._entities[entity_id] for entity_id in touched_entity_ids if entity_id in self._entities and self._entities[entity_id].memory_ids]
            )

    async def reset(self) -> None:
        entity_ids = list(self._entities.keys())
        self._entities.clear()
        self._episodes.clear()
        self._identity_keys.clear()
        self._entity_to_relations.clear()
        self._relations.clear()
        self._key_to_entities.clear()
        if self.entity_index is not None and entity_ids:
            await self.entity_index.delete_entities(entity_ids)

    async def stats(self) -> GraphStoreStats:
        return GraphStoreStats(
            backend="memory",
            memory_count=len({episode.memory_id for episode in self._episodes.values()}),
            entity_count=len(self._entities),
            relation_count=len(self._relations),
            episode_count=len(self._episodes),
            identity_key_count=len(self._identity_keys),
            cache_ready=True,
            cache_memory_count=len({episode.memory_id for episode in self._episodes.values()}),
            cache_entity_count=len(self._entities),
            cache_relation_count=len(self._relations),
            cache_episode_count=len(self._episodes),
            cache_build_ms=0.0,
        )

    async def passive_search(
        self,
        query_frame: GraphQueryFrame,
        *,
        max_entities: int = 5,
        max_relations: int = 8,
    ) -> GraphSearchResult:
        seed_entity_ids = await self._seed_entities(query_frame)
        relation_hits, relation_distances = self._rank_relations(
            query_frame,
            seed_entity_ids=seed_entity_ids,
            max_relations=max_relations,
            max_hops=1,
        )
        entity_ids = set(seed_entity_ids)
        for relation in relation_hits:
            entity_ids.add(relation.source_entity_id)
            entity_ids.add(relation.target_entity_id)

        entities = [self._entities[entity_id] for entity_id in list(entity_ids)[:max_entities]]
        episodes = self._episodes_for_relations(relation_hits)
        supporting_memory_ids = sorted(
            {memory_id for relation in relation_hits for memory_id in relation.memory_ids}
        )
        return GraphSearchResult(
            entities=entities,
            relations=relation_hits,
            episodes=episodes,
            seed_entity_ids=seed_entity_ids,
            relation_distances=relation_distances,
            supporting_memory_ids=supporting_memory_ids,
            search_plan=[
                "resolve exact identity keys from query",
                "score direct relations with one-hop expansion",
                "boost active, episode-backed, and intent-aligned relations",
            ],
        )

    async def traverse(
        self,
        query_frame: GraphQueryFrame,
        *,
        seed_entity_ids: list[str] | None = None,
        max_hops: int = 2,
        max_entities: int = 10,
        max_relations: int = 12,
    ) -> GraphSearchResult:
        seeds = seed_entity_ids or await self._seed_entities(query_frame)
        relation_hits, relation_distances = self._rank_relations(
            query_frame,
            seed_entity_ids=seeds,
            max_relations=max_relations,
            max_hops=max(1, max_hops),
        )
        entity_ids = set(seeds)
        for relation in relation_hits:
            entity_ids.add(relation.source_entity_id)
            entity_ids.add(relation.target_entity_id)

        entities = [self._entities[entity_id] for entity_id in list(entity_ids)[:max_entities]]
        episodes = self._episodes_for_relations(relation_hits)
        supporting_memory_ids = sorted(
            {memory_id for relation in relation_hits for memory_id in relation.memory_ids}
        )
        return GraphSearchResult(
            entities=entities,
            relations=relation_hits,
            episodes=episodes,
            seed_entity_ids=seeds,
            relation_distances=relation_distances,
            supporting_memory_ids=supporting_memory_ids,
            search_plan=[
                "resolve query seeds from exact identity keys and lexical matching",
                f"perform constrained traversal up to {max(1, max_hops)} hops",
                "rerank by lexical overlap, seed proximity, temporal validity, episode provenance, and relation intent",
            ],
        )

    async def get_entity(self, entity_id: str) -> GraphEntityNode | None:
        return self._entities.get(entity_id)

    async def get_episode(self, episode_id: str) -> GraphEpisode | None:
        return self._episodes.get(episode_id)

    async def find_facts(self, query: GraphFactQuery) -> list[GraphRelationEdge]:
        ranked = sorted(
            (
                relation
                for relation in self._relations.values()
                if self._relation_matches_fact_query(relation, query)
            ),
            key=lambda relation: self._score_fact_query_match(relation, query),
            reverse=True,
        )
        return ranked[: query.max_results]

    async def _upsert_entity(
        self,
        document: GraphDocument,
        entity,
    ) -> tuple[GraphEntityNode, list[IdentityResolutionResult]]:
        memory_id = document.memory_id
        identity_candidates = list(entity.identity_candidates)
        for raw_value in [entity.canonical_name, *entity.alias_values]:
            identity_candidates.append(
                GraphIdentityCandidate(
                    key_type=IdentityKeyType.NAME_VARIANT,
                    raw_value=raw_value,
                    confidence=0.6,
                )
            )
        keys = [
            build_identity_key(candidate, memory_id=memory_id)
            for candidate in identity_candidates
        ]
        resolution_events: list[IdentityResolutionResult] = []
        strong_matches: set[str] = set()
        weak_matches: set[str] = set()

        for key in keys:
            matches = self._key_to_entities.get(key.key_id, set())
            if not matches:
                continue
            if key.key_type in STRONG_KEY_TYPES:
                strong_matches.update(matches)
            else:
                weak_matches.update(matches)

        resolved_entity: GraphEntityNode
        if len(strong_matches) == 1:
            entity_id = next(iter(strong_matches))
            resolved_entity = self._entities[entity_id]
            resolution_events.append(
                IdentityResolutionResult(
                    status="exact_match",
                    entity_id=entity_id,
                    key=next((key for key in keys if key.key_type in STRONG_KEY_TYPES), None),
                )
            )
        elif len(strong_matches) > 1:
            resolved_entity = self._create_entity(memory_id, entity, provisional=True)
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
        elif entity_allows_name_auto_merge(entity.entity_type) and len(weak_matches) == 1:
            entity_id = next(iter(weak_matches))
            resolved_entity = self._entities[entity_id]
            resolution_events.append(
                IdentityResolutionResult(
                    status="exact_match",
                    entity_id=entity_id,
                )
            )
        else:
            similarity_hits = await self._similarity_hits(entity)
            decision = await self._resolve_ambiguous_entity(
                document=document,
                document_entity=entity,
                strong_match_ids=sorted(strong_matches),
                weak_match_ids=sorted(weak_matches),
                similarity_hits=similarity_hits,
            )
            if decision is None:
                resolved_entity = self._create_entity(memory_id, entity, provisional=False)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="created_new",
                        entity_id=resolved_entity.entity_id,
                    )
                )
            elif decision.status == "exact_match" and decision.entity_id in self._entities:
                resolved_entity = self._entities[decision.entity_id]
                resolution_events.append(decision)
            elif decision.status == "candidate_match":
                resolved_entity = self._create_entity(memory_id, entity, provisional=True)
                resolution_events.append(
                    decision.model_copy(update={"entity_id": resolved_entity.entity_id})
                )
            else:
                resolved_entity = self._create_entity(memory_id, entity, provisional=False)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="created_new",
                        entity_id=resolved_entity.entity_id,
                    )
                )

        self._merge_entity(memory_id, resolved_entity, entity, keys)
        return resolved_entity, resolution_events

    def _create_entity(self, memory_id: str, entity, *, provisional: bool) -> GraphEntityNode:
        graph_entity = GraphEntityNode(
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            alias_values=list(dict.fromkeys(entity.alias_values)),
            attributes=dict(entity.attributes),
            memory_ids=[memory_id],
            resolution_state="provisional" if provisional else "canonical",
        )
        self._entities[graph_entity.entity_id] = graph_entity
        return graph_entity

    def _merge_entity(
        self,
        memory_id: str,
        graph_entity: GraphEntityNode,
        document_entity,
        keys: list[GraphIdentityKey],
    ) -> None:
        graph_entity.updated_at = utc_now()
        graph_entity.canonical_name = graph_entity.canonical_name or document_entity.canonical_name
        graph_entity.alias_values = list(
            dict.fromkeys(
                [*graph_entity.alias_values, document_entity.canonical_name, *document_entity.alias_values]
            )
        )
        graph_entity.memory_ids = list(dict.fromkeys([*graph_entity.memory_ids, memory_id]))
        graph_entity.attributes.update(document_entity.attributes)
        if graph_entity.resolution_state == "provisional" and any(
            key.key_type in STRONG_KEY_TYPES for key in keys
        ):
            graph_entity.resolution_state = "canonical"

        for key in keys:
            existing_key = self._identity_keys.get(key.key_id)
            if existing_key is None:
                self._identity_keys[key.key_id] = key
                existing_key = key
            else:
                existing_key.raw_values = list(
                    dict.fromkeys([*existing_key.raw_values, *key.raw_values])
                )
                existing_key.last_seen_at = utc_now()
                existing_key.memory_ids = list(
                    dict.fromkeys([*existing_key.memory_ids, memory_id])
                )
                existing_key.confidence = max(existing_key.confidence, key.confidence)
            self._key_to_entities[existing_key.key_id].add(graph_entity.entity_id)
            if existing_key.key_id not in graph_entity.identity_key_ids:
                graph_entity.identity_key_ids.append(existing_key.key_id)

    def _upsert_episode(self, document: GraphDocument) -> GraphEpisode:
        if document.episode is not None:
            episode = document.episode
        else:
            episode = GraphEpisode(
                memory_id=document.memory_id,
                source_type="unknown",
                created_at=document.created_at,
            )
        episode.memory_id = document.memory_id
        if not episode.source_excerpt:
            episode.source_excerpt = _excerpt(document.source_text)
        if episode.extraction_confidence <= 0 and document.relations:
            episode.extraction_confidence = sum(
                max(0.0, min(relation.confidence, 1.0)) for relation in document.relations
            ) / len(document.relations)
        self._episodes[episode.episode_id] = episode
        return episode

    async def _upsert_relation(
        self,
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
        relation_id = self._deterministic_relation_id(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            fact=fact,
            valid_at=valid_at,
            invalid_at=invalid_at,
            expires_at=expires_at,
        )
        edge = self._relations.get(relation_id)
        if edge is None:
            pending = PendingFactContext(
                relation_type=relation_type,
                source_entity_id=source_entity_id,
                source_entity_name=self._entities[source_entity_id].canonical_name,
                target_entity_id=target_entity_id,
                target_entity_name=self._entities[target_entity_id].canonical_name,
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
                exact_duplicate.memory_ids = list(
                    dict.fromkeys([*exact_duplicate.memory_ids, memory_id])
                )
                exact_duplicate.episode_ids = list(
                    dict.fromkeys([*exact_duplicate.episode_ids, episode.episode_id])
                )
                exact_duplicate.updated_at = utc_now()
                exact_duplicate.confidence = max(exact_duplicate.confidence, confidence)
                return exact_duplicate, []

            action, resolved_edge, invalidated_ids = await self._maybe_invalidate_relations(
                document=document,
                episode=episode,
                pending=pending,
                candidates=candidates,
            )
            if action == "discard":
                return resolved_edge, invalidated_ids
            if action == "merge" and resolved_edge is not None:
                return resolved_edge, invalidated_ids
            edge = GraphRelationEdge(
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
            self._relations[relation_id] = edge
            self._entity_to_relations[source_entity_id].add(relation_id)
            self._entity_to_relations[target_entity_id].add(relation_id)
            return edge, invalidated_ids
        else:
            edge.memory_ids = list(dict.fromkeys([*edge.memory_ids, memory_id]))
            edge.episode_ids = list(dict.fromkeys([*edge.episode_ids, episode.episode_id]))
            edge.updated_at = utc_now()
            edge.confidence = max(edge.confidence, confidence)
            edge.valid_at = edge.valid_at or valid_at
            edge.invalid_at = invalid_at or edge.invalid_at
            edge.expires_at = expires_at or edge.expires_at
            return edge, []

    async def _maybe_invalidate_relations(
        self,
        *,
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
                relation = self._relations.get(relation_id)
                if relation is None or not self._is_active_relation(relation):
                    continue
                relation.invalidated_by_episode_id = episode.episode_id
                relation.updated_at = utc_now()
            return "create", None, deterministic_invalidated_ids

        candidate_contexts = [
            FactCandidateContext(
                relation_id=candidate.relation_id,
                relation_type=candidate.relation_type,
                source_entity_id=candidate.source_entity_id,
                source_entity_name=self._entities[candidate.source_entity_id].canonical_name,
                target_entity_id=candidate.target_entity_id,
                target_entity_name=self._entities[candidate.target_entity_id].canonical_name,
                fact=candidate.fact,
                confidence=candidate.confidence,
                memory_count=len(candidate.memory_ids),
                episode_count=len(candidate.episode_ids),
                valid_at=candidate.valid_at,
                invalid_at=candidate.invalid_at,
                expires_at=candidate.expires_at,
                active=self._is_active_relation(candidate),
                retrieval_reason="active_relation_family_candidate",
            )
            for candidate in candidates
        ]
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
            existing = self._relations.get(decision.chosen_relation_id)
            if existing is not None:
                existing.memory_ids = list(dict.fromkeys([*existing.memory_ids, document.memory_id]))
                existing.episode_ids = list(dict.fromkeys([*existing.episode_ids, episode.episode_id]))
                existing.updated_at = utc_now()
                return "merge", existing, []
        if decision.decision != "invalidate_existing":
            return "create", None, []

        invalidated_ids: list[str] = []
        for relation_id in decision.invalidated_relation_ids:
            relation = self._relations.get(relation_id)
            if relation is None or not self._is_active_relation(relation):
                continue
            relation.invalidated_by_episode_id = episode.episode_id
            relation.updated_at = utc_now()
            invalidated_ids.append(relation.relation_id)
        return "create", None, invalidated_ids

    async def _seed_entities(self, query_frame: GraphQueryFrame) -> list[str]:
        exact_entity_ids: set[str] = set()
        for candidate in query_frame.identity_candidates:
            exact_entity_ids.update(self._resolve_exact_entity_ids(candidate))

        if exact_entity_ids:
            return sorted(exact_entity_ids)

        score_map: dict[str, float] = {}
        for entity in self._entities.values():
            lexical_score = self._score_entity(query_frame, entity)
            if lexical_score > 0:
                score_map[entity.entity_id] = max(score_map.get(entity.entity_id, 0.0), lexical_score)

        if self.entity_index is not None:
            similarity_hits = await self.entity_index.search(query_frame.query, limit=6)
            for hit in similarity_hits:
                if hit.entity_id not in self._entities:
                    continue
                score_map[hit.entity_id] = max(score_map.get(hit.entity_id, 0.0), hit.score * 0.85)

        ranked_entity_ids = sorted(score_map, key=lambda entity_id: score_map[entity_id], reverse=True)
        return [
            entity_id
            for entity_id in ranked_entity_ids[:3]
            if entity_id in self._entities and score_map[entity_id] > 0
        ]

    def _resolve_exact_entity_ids(self, candidate: GraphIdentityCandidate) -> set[str]:
        try:
            key = build_identity_key(candidate)
        except ValueError:
            return set()
        return set(self._key_to_entities.get(key.key_id, set()))

    def _rank_relations(
        self,
        query_frame: GraphQueryFrame,
        *,
        seed_entity_ids: list[str],
        max_relations: int,
        max_hops: int,
    ) -> tuple[list[GraphRelationEdge], dict[str, int]]:
        allowed_relations = set(query_frame.allowed_relations)
        candidate_relation_ids: set[str] = set()
        relation_distances: dict[str, int] = {}

        if seed_entity_ids:
            frontier = deque((entity_id, 0) for entity_id in seed_entity_ids)
            visited_entity_depths = {entity_id: 0 for entity_id in seed_entity_ids}
            while frontier:
                entity_id, depth = frontier.popleft()
                for relation_id in self._entity_to_relations.get(entity_id, set()):
                    relation = self._relations[relation_id]
                    if relation.relation_type in allowed_relations:
                        candidate_relation_ids.add(relation_id)
                        relation_distances[relation_id] = min(
                            relation_distances.get(relation_id, depth + 1),
                            depth + 1,
                        )
                    if depth + 1 >= max_hops:
                        continue
                    next_entity_ids = {relation.source_entity_id, relation.target_entity_id} - {entity_id}
                    for next_entity_id in next_entity_ids:
                        next_depth = depth + 1
                        if visited_entity_depths.get(next_entity_id, next_depth + 1) <= next_depth:
                            continue
                        visited_entity_depths[next_entity_id] = next_depth
                        frontier.append((next_entity_id, next_depth))
        else:
            candidate_relation_ids.update(
                relation_id
                for relation_id, relation in self._relations.items()
                if relation.relation_type in allowed_relations
            )
            relation_distances.update({relation_id: 1 for relation_id in candidate_relation_ids})

        ranked = sorted(
            (
                self._relations[relation_id]
                for relation_id in candidate_relation_ids
                if not query_frame.prefer_current_state or self._is_active_relation(self._relations[relation_id])
            ),
            key=lambda relation: self._score_relation(query_frame, relation, seed_entity_ids),
            reverse=True,
        )
        selected_relations = [
            relation
            for relation in ranked[:max_relations]
            if self._score_relation(query_frame, relation, seed_entity_ids) > 0
        ]
        selected_distances = {
            relation.relation_id: relation_distances.get(relation.relation_id, 1)
            for relation in selected_relations
        }
        return selected_relations, selected_distances

    def _score_entity(self, query_frame: GraphQueryFrame, entity: GraphEntityNode) -> float:
        query_tokens = _tokenize(query_frame.query)
        if not query_tokens:
            return 0.0
        entity_text = " ".join([entity.canonical_name, *entity.alias_values])
        overlap = len(query_tokens & _tokenize(entity_text))
        if overlap == 0:
            return 0.0
        score = overlap / len(query_tokens)
        if entity.resolution_state == "canonical":
            score += 0.10
        return score

    def _score_relation(
        self,
        query_frame: GraphQueryFrame,
        relation: GraphRelationEdge,
        seed_entity_ids: list[str],
    ) -> float:
        query_tokens = _tokenize(query_frame.query)
        relation_text = " ".join(
            [
                relation.fact,
                relation.relation_type.value,
                self._entities[relation.source_entity_id].canonical_name,
                self._entities[relation.target_entity_id].canonical_name,
            ]
        )
        overlap = len(query_tokens & _tokenize(relation_text))
        score = overlap / max(len(query_tokens), 1)

        if relation.source_entity_id in seed_entity_ids or relation.target_entity_id in seed_entity_ids:
            score += 0.75
        if relation.relation_type in query_frame.allowed_relations:
            score += 0.15
        if self._is_active_relation(relation):
            score += 0.15 if query_frame.prefer_current_state else 0.05
        score += relation.confidence * 0.05
        return score

    @staticmethod
    def _is_active_relation(relation: GraphRelationEdge) -> bool:
        now = utc_now()
        if relation.invalidated_by_episode_id is not None:
            return False
        if relation.invalid_at and relation.invalid_at <= now:
            return False
        if relation.expires_at and relation.expires_at <= now:
            return False
        return True

    @staticmethod
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

    def _episodes_for_relations(self, relations: list[GraphRelationEdge]) -> list[GraphEpisode]:
        episode_ids = {
            episode_id
            for relation in relations
            for episode_id in [*relation.episode_ids, relation.invalidated_by_episode_id]
            if episode_id
        }
        return [self._episodes[episode_id] for episode_id in sorted(episode_ids) if episode_id in self._episodes]

    def _relation_matches_fact_query(
        self,
        relation: GraphRelationEdge,
        query: GraphFactQuery,
    ) -> bool:
        if query.active_only and not self._is_active_relation(relation):
            return False
        if query.relation_types and relation.relation_type not in set(query.relation_types):
            return False
        if query.source_entity_ids and relation.source_entity_id not in set(query.source_entity_ids):
            return False
        if query.target_entity_ids and relation.target_entity_id not in set(query.target_entity_ids):
            return False
        if query.anchor_entity_ids:
            anchor_ids = set(query.anchor_entity_ids)
            if relation.source_entity_id not in anchor_ids and relation.target_entity_id not in anchor_ids:
                return False
        if query.valid_at_or_after is not None and relation.invalid_at is not None and relation.invalid_at < query.valid_at_or_after:
            return False
        if query.valid_at_or_before is not None and relation.valid_at is not None and relation.valid_at > query.valid_at_or_before:
            return False
        return True

    def _score_fact_query_match(
        self,
        relation: GraphRelationEdge,
        query: GraphFactQuery,
    ) -> float:
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
        if self._is_active_relation(relation):
            score += 0.2
        score += min(max(relation.confidence, 0.0), 1.0) * 0.05
        score += len(relation.episode_ids) * 0.01
        return score

    async def close(self) -> None:
        if self.entity_index is not None:
            await self.entity_index.close()
        if self.adjudicator is not None:
            await self.adjudicator.close()
        if self.fact_adjudicator is not None:
            await self.fact_adjudicator.close()

    async def _similarity_hits(self, entity) -> list:
        if self.entity_index is None:
            return []
        return await self.entity_index.search(
            entity_similarity_text_parts(
                entity_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                alias_values=entity.alias_values,
                attributes=entity.attributes,
            ),
            entity_types=sorted(
                similarity_candidate_types(entity.entity_type),
                key=lambda value: value.value,
            ),
            limit=5,
        )

    async def _resolve_ambiguous_entity(
        self,
        *,
        document: GraphDocument,
        document_entity,
        strong_match_ids: list[str],
        weak_match_ids: list[str],
        similarity_hits: list,
    ) -> IdentityResolutionResult | None:
        candidate_contexts = self._build_candidate_contexts(
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
                        provenance_created_at=document.created_at.astimezone(timezone.utc),
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

    def _build_candidate_contexts(
        self,
        *,
        strong_match_ids: list[str],
        weak_match_ids: list[str],
        similarity_hits: list,
    ) -> list[EntityCandidateContext]:
        candidates: dict[str, EntityCandidateContext] = {}

        for entity_id in strong_match_ids:
            context = self._candidate_context_for_entity_id(
                entity_id,
                match_reason="multiple_strong_identity_matches",
            )
            if context is not None:
                candidates[entity_id] = context

        for entity_id in weak_match_ids:
            context = self._candidate_context_for_entity_id(
                entity_id,
                match_reason="weak_alias_only_match",
            )
            if context is not None:
                existing = candidates.get(entity_id)
                if existing is None:
                    candidates[entity_id] = context
                else:
                    existing.match_reasons = list(
                        dict.fromkeys([*existing.match_reasons, *context.match_reasons])
                    )

        for hit in similarity_hits:
            context = self._candidate_context_for_entity_id(
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

    def _candidate_context_for_entity_id(
        self,
        entity_id: str,
        *,
        match_reason: str,
        similarity_score: float | None = None,
    ) -> EntityCandidateContext | None:
        entity = self._entities.get(entity_id)
        if entity is None:
            return None
        relation_summaries = []
        for relation_id in list(self._entity_to_relations.get(entity_id, set()))[:4]:
            relation = self._relations.get(relation_id)
            if relation is None:
                continue
            counterpart_id = (
                relation.target_entity_id
                if relation.source_entity_id == entity_id
                else relation.source_entity_id
            )
            counterpart = self._entities.get(counterpart_id)
            counterpart_name = counterpart.canonical_name if counterpart is not None else counterpart_id
            relation_summaries.append(f"{relation.relation_type.value} -> {counterpart_name}")
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


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.casefold()) if token}


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
    pending_fact_key = relation_fact_signature_key(pending.relation_type, pending.fact)
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

        candidate_fact_key = relation_fact_signature_key(
            candidate.relation_type,
            candidate.fact,
        )
        if candidate_fact_key == pending_fact_key:
            continue

        invalidated_ids.append(candidate.relation_id)
    return invalidated_ids
