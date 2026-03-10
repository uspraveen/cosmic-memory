from fastapi.testclient import TestClient

from cosmic_memory.server.app import create_development_app


def test_health_endpoint():
    client = TestClient(create_development_app())
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "cosmic-memory"


def test_embedding_endpoint():
    client = TestClient(create_development_app())
    response = client.post(
        "/v1/embeddings/generate",
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

    block_response = client.get("/v1/core-facts")
    assert block_response.status_code == 200
    payload = block_response.json()
    assert len(payload["items"]) == 1
    assert "User prefers concise answers." in payload["rendered"]
