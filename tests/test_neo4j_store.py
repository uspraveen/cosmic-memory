import asyncio

from cosmic_memory.graph.neo4j_store import Neo4jGraphStore


class _FakeResult:
    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, statements):
        self._statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, statement, **_kwargs):
        self._statements.append(statement)
        return _FakeResult()


class _FakeDriver:
    def __init__(self, statements):
        self._statements = statements

    def session(self, database=None):
        return _FakeSession(self._statements)

    async def close(self):
        return None


def test_ensure_ready_seeds_relation_property_tokens():
    async def run():
        statements = []
        store = object.__new__(Neo4jGraphStore)
        store.driver = _FakeDriver(statements)
        store.database = "neo4j"
        store.entity_index = None
        store.adjudicator = None
        store.fact_adjudicator = None
        store._ready = False
        store._cache_lock = asyncio.Lock()
        store._search_cache = None
        store._cache_hydrated_at = None
        store._cache_build_ms = None

        await Neo4jGraphStore._ensure_ready(store)

        assert store._ready is True
        assert any(
            "GraphSchemaSeed" in statement
            and "invalid_at" in statement
            and "invalidated_by_episode_id" in statement
            for statement in statements
        )

    asyncio.run(run())
