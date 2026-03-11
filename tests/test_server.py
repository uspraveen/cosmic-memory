from fastapi.testclient import TestClient

from cosmic_memory.server.app import create_development_app


def test_health_endpoint():
    client = TestClient(create_development_app())
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "cosmic-memory"


def test_internal_token_protects_service_endpoints(monkeypatch):
    monkeypatch.setenv("GATEWAY_INTERNAL_TOKEN", "internal-secret")
    client = TestClient(create_development_app())

    unauthorized = client.get("/v1/core-facts")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/v1/core-facts",
        headers={"X-Internal-Token": "internal-secret"},
    )
    assert authorized.status_code == 200


def test_health_endpoint_stays_public_with_internal_token(monkeypatch):
    monkeypatch.setenv("GATEWAY_INTERNAL_TOKEN", "internal-secret")
    client = TestClient(create_development_app())

    response = client.get("/health")
    assert response.status_code == 200


def test_embedding_endpoint():
    client = TestClient(create_development_app())
    response = client.post(
        "/v1/embeddings/generate",
        headers={"X-Internal-Token": ""},
        json={
            "texts": ["Cosmic memory", "Passive recall"],
            "dimensions": 128,
            "batch_size": 2,
            "max_parallel_requests": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "hash-embedding-dev"
    assert payload["dimensions"] == 128
    assert len(payload["items"]) == 2
    assert len(payload["items"][0]["vector"]) == 128


def test_core_fact_endpoint():
    client = TestClient(create_development_app())
    write_response = client.post(
        "/v1/core-facts",
        headers={"X-Internal-Token": ""},
        json={
            "title": "Preference",
            "fact": "User prefers concise answers.",
            "canonical_key": "preferences.response_style",
            "priority": 300,
            "always_include": True,
            "tags": ["preference"],
            "metadata": {},
            "provenance": {
                "source_kind": "gateway",
                "created_by": "test",
            },
        },
    )
    assert write_response.status_code == 201

    block_response = client.get("/v1/core-facts", headers={"X-Internal-Token": ""})
    assert block_response.status_code == 200
    payload = block_response.json()
    assert len(payload["items"]) == 1
    assert "User prefers concise answers." in payload["rendered"]


def test_episode_ingest_endpoint():
    client = TestClient(create_development_app())
    response = client.post(
        "/v1/episodes",
        headers={"X-Internal-Token": ""},
        json={
            "observations": [
                {"role": "user", "content": "We should improve graph retrieval."},
                {"role": "assistant", "content": "I will add retrieval recipes."},
            ],
            "provenance": {
                "source_kind": "gateway",
                "created_by": "test",
                "session_id": "sess_20260311",
            },
            "extract_graph": False,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["observation_count"] == 2
    assert payload["record"]["kind"] == "transcript"
    assert payload["record"]["metadata"]["episode_observation_count"] == 2
    assert payload["graph_episode_id"] is None


def test_agent_surface_endpoints():
    client = TestClient(create_development_app())

    schema_response = client.get("/v1/agent/schema-context", headers={"X-Internal-Token": ""})
    assert schema_response.status_code == 200
    schema_payload = schema_response.json()
    assert any(tool["name"] == "passive_search" for tool in schema_payload["tools"])

    plan_response = client.post(
        "/v1/agent/plan",
        headers={"X-Internal-Token": ""},
        json={"query": "What is currently blocking Cosmic Memory?"},
    )
    assert plan_response.status_code == 200
    plan_payload = plan_response.json()
    assert plan_payload["recommended_mode"] in {"hybrid", "active"}
    assert "passive_search" in plan_payload["tool_sequence"]

    brief_response = client.post(
        "/v1/agent/memory-brief",
        headers={"X-Internal-Token": ""},
        json={"query": "What is currently blocking Cosmic Memory?"},
    )
    assert brief_response.status_code == 200
    brief_payload = brief_response.json()
    assert "plan" in brief_payload
    assert "passive" in brief_payload
