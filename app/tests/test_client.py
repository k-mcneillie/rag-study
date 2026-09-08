"""``RagApiClient`` speaks the service's wire contract.

The service is replaced by an :class:`httpx.MockTransport`, so these tests
need nothing running.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag_chat.client import ApiError, RagApiClient
from rag_chat.models import (
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    NoContextEvent,
    PromptId,
)

Handler = Callable[[httpx.Request], httpx.Response]

_SSE = (
    "event: citations\n"
    'data: {"items": [{"position": 1, "document_id": "d1", "page_number": 4, '
    '"section": "1 Intro", "score": 0.7}]}\n\n'
    'event: delta\ndata: {"reasoning": true, "text": "hmm "}\n\n'
    'event: delta\ndata: {"reasoning": false, "text": "the answer [1]"}\n\n'
    'event: done\ndata: {"model_name": "m", '
    '"prompt": {"name": "p", "version": "v2"}}\n\n'
)


def _client(handler: Handler, *, api_key: str = "k") -> RagApiClient:
    """Build a client whose transport is the given handler.

    Args:
        handler: Responds to each request.
        api_key: The key the client should send.

    Returns:
        A client wired to the handler.
    """
    return RagApiClient("http://svc", api_key, transport=httpx.MockTransport(handler))


async def test_answer_stream_parses_events_and_sends_the_key() -> None:
    """The stream is decoded into typed events and carries ``X-API-Key``."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=_SSE
        )

    events = [event async for event in _client(handler).answer_stream("q", top_k=3)]

    assert seen["key"] == "k"
    assert seen["body"] == {"query": "q", "top_k": 3}
    assert isinstance(events[0], CitationsEvent)
    assert events[0].items[0].document_id == "d1"
    assert events[1] == DeltaEvent(text="hmm ", reasoning=True)
    assert events[2] == DeltaEvent(text="the answer [1]", reasoning=False)
    assert events[3] == DoneEvent(model_name="m", prompt=PromptId("p", "v2"))


async def test_answer_stream_raises_apierror_on_422() -> None:
    """A 422 is surfaced as ``ApiError`` carrying the service's detail."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Query must not be empty."})

    with pytest.raises(ApiError) as excinfo:
        async for _ in _client(handler).answer_stream("  "):
            pass

    assert excinfo.value.status_code == 422
    assert "must not be empty" in str(excinfo.value)


async def test_no_context_is_one_event() -> None:
    """A ``no_context`` stream decodes to a single event."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="event: no_context\ndata: {}\n\n",
        )

    events = [event async for event in _client(handler).answer_stream("q")]

    assert events == [NoContextEvent()]


async def test_feedback_posts_the_expected_shape() -> None:
    """The feedback body matches the contract and carries the key."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("x-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(204)

    await _client(handler).feedback(
        vote="down",
        query="q",
        answer="a",
        model_name="m",
        prompt=PromptId("p", "v1"),
        citations=(),
    )

    assert captured["key"] == "k"
    assert captured["body"] == {
        "vote": "down",
        "query": "q",
        "answer": "a",
        "model_name": "m",
        "prompt": {"name": "p", "version": "v1"},
        "citations": [],
    }


async def test_health_is_decoded() -> None:
    """``/health`` becomes a ``HealthInfo``."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model": "m",
                "provider": "ollama",
                "reranker": "vector-only",
                "prompt": "p/v1",
                "top_k": 5,
            },
        )

    info = await _client(handler).health()

    assert info.model == "m"
    assert info.reranker == "vector-only"
    assert info.top_k == 5


async def test_no_key_means_no_header() -> None:
    """An empty key is not sent as a blank header."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["has_key"] = "x-api-key" in request.headers
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model": "m",
                "provider": "o",
                "reranker": "vector-only",
                "prompt": "p/v1",
                "top_k": 5,
            },
        )

    await _client(handler, api_key="").health()

    assert captured["has_key"] is False
