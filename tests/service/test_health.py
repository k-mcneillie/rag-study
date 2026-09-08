"""``GET /health`` reports the configuration and needs no key."""

from __future__ import annotations

from tests.service.conftest import Harness


def test_health_is_open_and_describes_the_configuration(harness: Harness) -> None:
    """The health check answers without a key and names the model.

    Args:
        harness: The test app.
    """
    response = harness.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-model"
    assert body["provider"] == "ollama"
    assert body["reranker"] == "vector-only"
    assert body["prompt"] == "retrieval_context/v1"
    assert body["top_k"] == 5
