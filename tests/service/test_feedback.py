"""``POST /feedback`` appends one JSON line and returns 204."""

from __future__ import annotations

import json

from tests.service.conftest import API_KEY, Harness

_AUTH = {"X-API-Key": API_KEY}

_BODY = {
    "vote": "up",
    "query": "How is the sample defined?",
    "answer": "Part-time staff were excluded [1].",
    "model_name": "fake-model",
    "prompt": {"name": "retrieval_context", "version": "v1"},
    "citations": [
        {
            "position": 1,
            "document_id": "doc-1",
            "page_number": 12,
            "section": "2 Methods",
            "score": 0.88,
        }
    ],
}


def test_a_rating_is_written_to_the_log(harness: Harness) -> None:
    """The posted fields land in the feedback file as one line.

    Args:
        harness: The test app.
    """
    response = harness.client.post("/feedback", json=_BODY, headers=_AUTH)

    assert response.status_code == 204

    lines = harness.feedback_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["vote"] == "up"
    assert record["query"] == _BODY["query"]
    assert record["answer"] == _BODY["answer"]
    assert record["model_name"] == "fake-model"
    assert record["prompt_name"] == "retrieval_context"
    assert record["prompt_version"] == "v1"
    assert record["citations"] == [
        {
            "position": 1,
            "document_id": "doc-1",
            "page_number": 12,
            "section": "2 Methods",
            "score": 0.88,
        }
    ]
    assert "recorded_at" in record


def test_an_unknown_vote_is_rejected(harness: Harness) -> None:
    """Only ``up`` and ``down`` are accepted.

    Args:
        harness: The test app.
    """
    response = harness.client.post(
        "/feedback", json={**_BODY, "vote": "maybe"}, headers=_AUTH
    )

    assert response.status_code == 422


def test_citations_are_optional(harness: Harness) -> None:
    """A rating with no citations is still recorded.

    Args:
        harness: The test app.
    """
    response = harness.client.post(
        "/feedback", json={**_BODY, "citations": []}, headers=_AUTH
    )

    assert response.status_code == 204
    record = json.loads(harness.feedback_path.read_text(encoding="utf-8"))
    assert record["citations"] == []
