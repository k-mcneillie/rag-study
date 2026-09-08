"""``POST /answer`` streams citations, deltas, and a terminal event."""

from __future__ import annotations

import json

from rag.domain.models import AnswerDelta
from tests.service.conftest import API_KEY, Harness, make_context, parse_sse

_AUTH = {"X-API-Key": API_KEY}


def _events(harness: Harness, **payload: object) -> list[tuple[str, str]]:
    """POST a question and return the parsed SSE events.

    Args:
        harness: The test app.
        payload: The JSON request body.

    Returns:
        The ``(event, data)`` pairs streamed back.
    """
    response = harness.client.post("/answer", json=payload, headers=_AUTH)
    assert response.status_code == 200
    return parse_sse(response.text)


def test_a_successful_answer_streams_citations_then_deltas_then_done(
    harness: Harness,
) -> None:
    """The happy path is ``citations`` → ``delta``* → ``done``.

    Args:
        harness: The test app.
    """
    events = _events(harness, query="what is the answer?")
    names = [name for name, _ in events]

    assert names[0] == "citations"
    assert names[-1] == "done"
    assert names.count("delta") == 2
    assert set(names[1:-1]) == {"delta"}

    citations = json.loads(events[0][1])
    assert [item["position"] for item in citations["items"]] == [1, 2]
    assert citations["items"][0]["document_id"] == "doc-1"
    assert "text" not in citations["items"][0]

    reasoning, answer = (json.loads(data) for name, data in events if name == "delta")
    assert reasoning == {"reasoning": True, "text": "Thinking. "}
    assert answer == {"reasoning": False, "text": "The answer is 42 [1]."}

    done = json.loads(events[-1][1])
    assert done == {
        "model_name": "fake-model",
        "prompt": {"name": "retrieval_context", "version": "v1"},
    }


def test_top_k_defaults_to_the_service_setting(harness: Harness) -> None:
    """An omitted ``top_k`` uses the configured default.

    Args:
        harness: The test app.
    """
    _events(harness, query="q")
    _events(harness, query="q", top_k=3)

    assert harness.orchestrator.calls == [("q", 5), ("q", 3)]


def test_no_matching_context_ends_the_stream_without_calling_the_model(
    harness: Harness,
) -> None:
    """An empty retrieval yields a single ``no_context`` event.

    Args:
        harness: The test app.
    """
    harness.orchestrator.context = make_context(chunks=0)

    events = _events(harness, query="nothing matches this")

    assert [name for name, _ in events] == ["no_context"]


def test_generation_failure_becomes_an_error_event(harness: Harness) -> None:
    """A ``GenerationError`` is reported as an ``error`` event, not a 500.

    Args:
        harness: The test app.
    """
    harness.chat_model.error = "the model service is unreachable"

    events = _events(harness, query="q")
    names = [name for name, _ in events]

    assert names == ["citations", "error"]
    assert json.loads(events[-1][1]) == {"message": "the model service is unreachable"}


def test_an_empty_query_is_rejected_before_the_stream(harness: Harness) -> None:
    """A blank query returns 422 rather than opening a stream.

    Args:
        harness: The test app.
    """
    harness.orchestrator.error = "Query must not be empty."

    response = harness.client.post("/answer", json={"query": "   "}, headers=_AUTH)

    assert response.status_code == 422
    assert response.json()["detail"] == "Query must not be empty."


def test_a_model_that_only_reasons_still_finishes_with_done(harness: Harness) -> None:
    """A stream of pure reasoning still ends on ``done``.

    Args:
        harness: The test app.
    """
    harness.chat_model.deltas = (AnswerDelta(text="hmm", reasoning=True),)

    events = _events(harness, query="q")

    assert [name for name, _ in events] == ["citations", "delta", "done"]
