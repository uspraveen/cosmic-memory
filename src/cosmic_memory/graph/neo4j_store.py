"""Experimental persistent Neo4j graph store."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime

from cosmic_memory.domain.models import utc_now
from cosmic_memory.graph.dev_store import InMemoryGraphStore
from cosmic_memory.graph.identity import build_identity_key
from cosmic_memory.graph.models import (
    GraphDocument,
    GraphEntityNode,
    GraphIdentityCandidate,
    GraphIdentityKey,
    GraphIngestResult,
    GraphRelationEdge,
    GraphSearchResult,
    IdentityResolutionCandidate,
    IdentityResolutionResult,
)
from cosmic_memory.graph.ontology import IdentityKeyType, RelationType

STRONG_KEY_TYPES = {
    IdentityKeyType.EMAIL,
    IdentityKeyType.PHONE,
    IdentityKeyType.EXTERNAL_ACCOUNT,
}


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
        self._ready = False
        self._cache_lock = asyncio.Lock()
        self._search_cache: InMemoryGraphStore | None = None

    async def ingest_document(self, document: GraphDocument) -> GraphIngestResult:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
            ref_to_entity_id: dict[str, str] = {}
            resolution_events: list[IdentityResolutionResult] = []

            for entity in document.entities:
                resolved_entity, events = await self._upsert_entity(
                    session,
                    memory_id=document.memory_id,
                    document_entity=entity,
                )
                ref_to_entity_id[entity.local_ref] = resolved_entity.entity_id
                resolution_events.extend(events)

            relation_ids: list[str] = []
            for relation in document.relations:
                source_entity_id = ref_to_entity_id.get(relation.source_ref)
                target_entity_id = ref_to_entity_id.get(relation.target_ref)
                if source_entity_id is None or target_entity_id is None:
                    continue
                edge = await self._upsert_relation(
                    session,
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

        result = GraphIngestResult(
            memory_id=document.memory_id,
            entity_ids=sorted(set(ref_to_entity_id.values())),
            relation_ids=sorted(set(relation_ids)),
            resolution_events=resolution_events,
        )
        await self._cache_ingest_document(document)
        return result

    async def remove_memory(self, memory_id: str) -> None:
        await self._ensure_ready()
        async with self.driver.session(database=self.database) as session:
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
                            rel.updated_at = $updated_at
                        """,
                        relation_id=row["relation_id"],
                        memory_ids=remaining,
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
        await self._cache_remove_memory(memory_id)

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

    async def close(self) -> None:
        await self.driver.close()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        async with self.driver.session(database=self.database) as session:
            for statement in (
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
                "CREATE CONSTRAINT identity_key_unique IF NOT EXISTS FOR (k:IdentityKey) REQUIRE k.key_id IS UNIQUE",
                "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR (r:Relation) REQUIRE r.relation_id IS UNIQUE",
                "CREATE INDEX identity_key_type_idx IF NOT EXISTS FOR (k:IdentityKey) ON (k.key_type)",
                "CREATE INDEX relation_type_idx IF NOT EXISTS FOR (r:Relation) ON (r.relation_type)",
            ):
                result = await session.run(statement)
                await result.consume()
        self._ready = True

    async def _upsert_entity(self, session, *, memory_id: str, document_entity) -> tuple[GraphEntityNode, list[IdentityResolutionResult]]:
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
        elif weak_matches:
            resolved_entity = _new_entity(memory_id, document_entity, provisional=True)
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
        relation_id = _deterministic_relation_id(
            memory_id=memory_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            fact=fact,
        )
        existing = await self._get_relation(session, relation_id)
        if existing is None:
            relation = GraphRelationEdge(
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
        else:
            relation = existing
            relation.memory_ids = list(dict.fromkeys([*relation.memory_ids, memory_id]))
            relation.updated_at = utc_now()
            relation.confidence = max(relation.confidence, confidence)
            relation.valid_at = relation.valid_at or valid_at
            relation.invalid_at = invalid_at or relation.invalid_at
            relation.expires_at = expires_at or relation.expires_at

        await self._persist_relation(session, relation)
        return relation

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

    async def _persist_relation(self, session, relation: GraphRelationEdge) -> None:
        await session.run(
            """
            MERGE (rel:Relation {relation_id: $relation_id})
            SET rel.relation_type = $relation_type,
                rel.fact = $fact,
                rel.memory_ids = $memory_ids,
                rel.confidence = $confidence,
                rel.created_at = coalesce(rel.created_at, $created_at),
                rel.updated_at = $updated_at,
                rel.valid_at = $valid_at,
                rel.invalid_at = $invalid_at,
                rel.expires_at = $expires_at,
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
            confidence=relation.confidence,
            created_at=_serialize_datetime(relation.created_at),
            updated_at=_serialize_datetime(relation.updated_at),
            valid_at=_serialize_datetime(relation.valid_at),
            invalid_at=_serialize_datetime(relation.invalid_at),
            expires_at=_serialize_datetime(relation.expires_at),
            source_entity_id=relation.source_entity_id,
            target_entity_id=relation.target_entity_id,
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
        store = InMemoryGraphStore()
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

        store._entities = entities
        store._identity_keys = identity_keys
        store._relations = relations
        store._key_to_entities = key_to_entities
        store._entity_to_relations = entity_to_relations
        return store

    async def _cache_ingest_document(self, document: GraphDocument) -> None:
        async with self._cache_lock:
            if self._search_cache is None:
                self._search_cache = await self._hydrate_search_store()
                return
            await self._search_cache.ingest_document(document)

    async def _cache_remove_memory(self, memory_id: str) -> None:
        async with self._cache_lock:
            if self._search_cache is None:
                return
            await self._search_cache.remove_memory(memory_id)


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
        confidence=float(props.get("confidence") or 1.0),
        created_at=_deserialize_datetime(props.get("created_at")) or utc_now(),
        updated_at=_deserialize_datetime(props.get("updated_at")) or utc_now(),
        valid_at=_deserialize_datetime(props.get("valid_at")),
        invalid_at=_deserialize_datetime(props.get("invalid_at")),
        expires_at=_deserialize_datetime(props.get("expires_at")),
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
    memory_id: str,
    source_entity_id: str,
    target_entity_id: str,
    relation_type: RelationType,
    fact: str,
) -> str:
    payload = "||".join([memory_id, source_entity_id, target_entity_id, relation_type.value, fact])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"rel_{digest[:24]}"
