"""In-memory graph store for local development and contract testing."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque

from cosmic_memory.domain.models import utc_now
from cosmic_memory.graph.entity_index import EntitySimilarityIndex, entity_similarity_text_parts
from cosmic_memory.graph.identity import build_identity_key
from cosmic_memory.graph.models import (
    GraphDocument,
    GraphEntityNode,
    GraphIdentityCandidate,
    GraphIdentityKey,
    GraphIngestResult,
    GraphQueryFrame,
    GraphRelationEdge,
    GraphSearchResult,
    IdentityResolutionCandidate,
    IdentityResolutionResult,
)
from cosmic_memory.graph.ontology import IdentityKeyType, RelationType
from cosmic_memory.graph.resolution import (
    STRONG_KEY_TYPES,
    entity_allows_name_auto_merge,
    similarity_candidate_types,
)


class InMemoryGraphStore:
    """Small in-memory graph store with strict identity merge rules."""

    def __init__(self, *, entity_index: EntitySimilarityIndex | None = None) -> None:
        self._entities: dict[str, GraphEntityNode] = {}
        self._identity_keys: dict[str, GraphIdentityKey] = {}
        self._entity_to_relations: dict[str, set[str]] = defaultdict(set)
        self._relations: dict[str, GraphRelationEdge] = {}
        self._key_to_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_index = entity_index

    async def ingest_document(self, document: GraphDocument) -> GraphIngestResult:
        ref_to_entity_id: dict[str, str] = {}
        resolution_events: list[IdentityResolutionResult] = []
        changed_entity_ids: set[str] = set()

        for entity in document.entities:
            resolved_entity, events = await self._upsert_entity(document.memory_id, entity)
            ref_to_entity_id[entity.local_ref] = resolved_entity.entity_id
            changed_entity_ids.add(resolved_entity.entity_id)
            resolution_events.extend(events)

        relation_ids: list[str] = []
        for relation in document.relations:
            source_entity_id = ref_to_entity_id.get(relation.source_ref)
            target_entity_id = ref_to_entity_id.get(relation.target_ref)
            if source_entity_id is None or target_entity_id is None:
                continue
            edge = self._upsert_relation(
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
            relation_ids.append(edge.relation_id)

        if self.entity_index is not None and changed_entity_ids:
            await self.entity_index.sync_entities(
                [self._entities[entity_id] for entity_id in changed_entity_ids if entity_id in self._entities]
            )

        return GraphIngestResult(
            memory_id=document.memory_id,
            entity_ids=sorted(set(ref_to_entity_id.values())),
            relation_ids=sorted(set(relation_ids)),
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
        relation_ids_to_delete: list[str] = []
        touched_entity_ids: set[str] = set()
        for relation_id, relation in self._relations.items():
            if memory_id not in relation.memory_ids:
                continue
            relation.memory_ids = [value for value in relation.memory_ids if value != memory_id]
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

        if self.entity_index is not None:
            await self.entity_index.delete_entities(
                [entity_id for entity_id in touched_entity_ids if entity_id in self._entities and not self._entities[entity_id].memory_ids]
            )
            await self.entity_index.sync_entities(
                [self._entities[entity_id] for entity_id in touched_entity_ids if entity_id in self._entities and self._entities[entity_id].memory_ids]
            )

    async def passive_search(
        self,
        query_frame: GraphQueryFrame,
        *,
        max_entities: int = 5,
        max_relations: int = 8,
    ) -> GraphSearchResult:
        seed_entity_ids = await self._seed_entities(query_frame)
        relation_hits = self._rank_relations(
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
        supporting_memory_ids = sorted(
            {memory_id for relation in relation_hits for memory_id in relation.memory_ids}
        )
        return GraphSearchResult(
            entities=entities,
            relations=relation_hits,
            supporting_memory_ids=supporting_memory_ids,
            search_plan=[
                "resolve exact identity keys from query",
                "score direct relations with one-hop expansion",
                "boost active and intent-aligned relations",
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
        relation_hits = self._rank_relations(
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
        supporting_memory_ids = sorted(
            {memory_id for relation in relation_hits for memory_id in relation.memory_ids}
        )
        return GraphSearchResult(
            entities=entities,
            relations=relation_hits,
            supporting_memory_ids=supporting_memory_ids,
            search_plan=[
                "resolve query seeds from exact identity keys and lexical matching",
                f"perform constrained traversal up to {max(1, max_hops)} hops",
                "rerank by lexical overlap, seed proximity, temporal validity, and relation intent",
            ],
        )

    async def get_entity(self, entity_id: str) -> GraphEntityNode | None:
        return self._entities.get(entity_id)

    async def _upsert_entity(self, memory_id: str, entity) -> tuple[GraphEntityNode, list[IdentityResolutionResult]]:
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
        elif weak_matches:
            resolved_entity = self._create_entity(memory_id, entity, provisional=True)
            resolution_events.append(
                IdentityResolutionResult(
                    status="candidate_match",
                    entity_id=resolved_entity.entity_id,
                    candidates=[
                        IdentityResolutionCandidate(
                            entity_id=entity_id,
                            reason="weak_alias_only_match",
                            confidence=0.35,
                        )
                        for entity_id in sorted(weak_matches)
                    ],
                )
            )
        elif self.entity_index is not None:
            similarity_hits = await self.entity_index.search(
                entity_similarity_text_parts(
                    entity_type=entity.entity_type,
                    canonical_name=entity.canonical_name,
                    alias_values=entity.alias_values,
                    attributes=entity.attributes,
                ),
                entity_types=sorted(similarity_candidate_types(entity.entity_type), key=lambda value: value.value),
                limit=5,
            )
            auto_merge_hits = [hit for hit in similarity_hits if hit.score >= 0.92]
            candidate_hits = [hit for hit in similarity_hits if 0.78 <= hit.score < 0.92]
            if auto_merge_hits:
                resolved_entity = self._entities[auto_merge_hits[0].entity_id]
                resolution_events.append(
                    IdentityResolutionResult(
                        status="exact_match",
                        entity_id=resolved_entity.entity_id,
                    )
                )
            elif candidate_hits:
                resolved_entity = self._create_entity(memory_id, entity, provisional=True)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="candidate_match",
                        entity_id=resolved_entity.entity_id,
                        candidates=[
                            IdentityResolutionCandidate(
                                entity_id=hit.entity_id,
                                reason="entity_similarity_candidate",
                                confidence=min(max(hit.score, 0.0), 1.0),
                            )
                            for hit in candidate_hits
                        ],
                    )
                )
            else:
                resolved_entity = self._create_entity(memory_id, entity, provisional=False)
                resolution_events.append(
                    IdentityResolutionResult(
                        status="created_new",
                        entity_id=resolved_entity.entity_id,
                    )
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

    def _upsert_relation(
        self,
        *,
        memory_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: RelationType,
        fact: str,
        confidence: float,
        valid_at,
        invalid_at,
        expires_at,
    ) -> GraphRelationEdge:
        relation_id = self._deterministic_relation_id(
            memory_id=memory_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            fact=fact,
        )
        edge = self._relations.get(relation_id)
        if edge is None:
            edge = GraphRelationEdge(
                relation_id=relation_id,
                relation_type=relation_type,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                fact=fact,
                memory_ids=[memory_id],
                confidence=confidence,
                valid_at=valid_at,
                invalid_at=invalid_at,
                expires_at=expires_at,
            )
            self._relations[relation_id] = edge
        else:
            edge.memory_ids = list(dict.fromkeys([*edge.memory_ids, memory_id]))
            edge.updated_at = utc_now()
            edge.confidence = max(edge.confidence, confidence)
            edge.valid_at = edge.valid_at or valid_at
            edge.invalid_at = invalid_at or edge.invalid_at
            edge.expires_at = expires_at or edge.expires_at

        self._entity_to_relations[source_entity_id].add(relation_id)
        self._entity_to_relations[target_entity_id].add(relation_id)
        return edge

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
                score_map[hit.entity_id] = max(score_map.get(hit.entity_id, 0.0), hit.score * 0.85)

        ranked_entity_ids = sorted(score_map, key=lambda entity_id: score_map[entity_id], reverse=True)
        return [entity_id for entity_id in ranked_entity_ids[:3] if score_map[entity_id] > 0]

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
    ) -> list[GraphRelationEdge]:
        allowed_relations = set(query_frame.allowed_relations)
        candidate_relation_ids: set[str] = set()

        if seed_entity_ids:
            frontier = deque((entity_id, 0) for entity_id in seed_entity_ids)
            visited_entities = set(seed_entity_ids)
            while frontier:
                entity_id, depth = frontier.popleft()
                for relation_id in self._entity_to_relations.get(entity_id, set()):
                    relation = self._relations[relation_id]
                    if relation.relation_type in allowed_relations:
                        candidate_relation_ids.add(relation_id)
                    if depth + 1 >= max_hops:
                        continue
                    next_entity_ids = {relation.source_entity_id, relation.target_entity_id} - {entity_id}
                    for next_entity_id in next_entity_ids:
                        if next_entity_id in visited_entities:
                            continue
                        visited_entities.add(next_entity_id)
                        frontier.append((next_entity_id, depth + 1))
        else:
            candidate_relation_ids.update(
                relation_id
                for relation_id, relation in self._relations.items()
                if relation.relation_type in allowed_relations
            )

        ranked = sorted(
            (self._relations[relation_id] for relation_id in candidate_relation_ids),
            key=lambda relation: self._score_relation(query_frame, relation, seed_entity_ids),
            reverse=True,
        )
        return [
            relation
            for relation in ranked[:max_relations]
            if self._score_relation(query_frame, relation, seed_entity_ids) > 0
        ]

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
        if relation.invalid_at and relation.invalid_at <= now:
            return False
        if relation.expires_at and relation.expires_at <= now:
            return False
        return True

    @staticmethod
    def _deterministic_relation_id(
        *,
        memory_id: str,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: RelationType,
        fact: str,
    ) -> str:
        payload = "||".join([memory_id, source_entity_id, target_entity_id, relation_type.value, fact])
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"rel_{digest[:24]}"

    async def close(self) -> None:
        if self.entity_index is not None:
            await self.entity_index.close()


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.casefold()) if token}
