"""When ``RAG_API_KEY`` is set, every route except ``/health`` requires it."""

from __future__ import annotations

import pytest

from tests.service.conftest import API_KEY, Harness


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/answer"), ("post", "/feedback"), ("post", "/documents")],
)
def test_data_routes_reject_a_missing_key(
    harness: Harness, method: str, path: str
) -> None:
    """A request with no ``X-API-Key`` header is refused.

    Args:
        harness: The test app.
        method: The HTTP method to use.
        path: The route to call.
    """
    response = getattr(harness.client, method)(path, json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing or invalid API key."


def test_a_wrong_key_is_refused(harness: Harness) -> None:
    """A request with the wrong key is refused.

    Args:
        harness: The test app.
    """
    response = harness.client.post(
        "/answer", json={"query": "hi"}, headers={"X-API-Key": "not-the-key"}
    )

    assert response.status_code == 401


def test_the_right_key_is_accepted(harness: Harness) -> None:
    """A request carrying the configured key is let through.

    Args:
        harness: The test app.
    """
    response = harness.client.post(
        "/answer", json={"query": "hi"}, headers={"X-API-Key": API_KEY}
    )

    assert response.status_code == 200


def test_an_unset_key_leaves_the_service_open(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ``RAG_API_KEY`` configured, a keyless request is allowed.

    Args:
        harness: The test app.
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    response = harness.client.post("/answer", json={"query": "hi"})

    assert response.status_code == 200
