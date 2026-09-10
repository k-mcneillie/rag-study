"""``OpenAIRagClient`` and ``_StreamDecoder`` speak the OpenAI wire dialect.

The provider is replaced by an :class:`httpx.MockTransport`, so these tests need
nothing running.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from rag_chat_openai.client import ApiError, OpenAIRagClient, _StreamDecoder
from rag_chat_openai.models import Citation, Event

Handler = Callable[[httpx.Request], httpx.Response]


def _sse(*payloads: str) -> str:
    """Frame raw ``data:`` payloads as an SSE body.

    Args:
        *payloads: The text after ``data:`` for each event.

    Returns:
        The concatenated ``text/event-stream`` body.
    """
    return "".join(f"data: {payload}\n\n" for payload in payloads)


def _stream_response(body: str) -> httpx.Response:
    """Build a streamed SSE response.

    Args:
        body: The full event-stream body.

    Returns:
        A 200 response with the event-stream content type.
    """
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)


def _client(
    handler: Handler, *, api_key: str = "k", **kwargs: object
) -> OpenAIRagClient:
    """Build a client whose transport is the given handler.

    Args:
        handler: Responds to each request.
        api_key: The bearer token the client should send.
        **kwargs: Passed through to :class:`OpenAIRagClient`.

    Returns:
        A client wired to the handler.
    """
    return OpenAIRagClient(
        "http://provider",
        api_key,
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_answer_stream_decodes_content_and_sends_a_bearer_token() -> None:
    """Content deltas become ``answer`` events and the request carries the token."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return _stream_response(
            _sse(
                json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
                json.dumps({"choices": [{"delta": {"content": "lo"}}], "model": "m"}),
                json.dumps(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "m"}
                ),
                "[DONE]",
            )
        )

    decoder = _StreamDecoder()
    events = [event async for event in _client(handler).answer_stream("q", decoder)]

    assert seen["auth"] == "Bearer k"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"] == {
        "model": "",
        "messages": [{"role": "user", "content": "q"}],
        "stream": True,
    }
    assert events == [
        Event("answer", "Hel"),
        Event("answer", "lo"),
        Event("done", "m"),
    ]
    assert decoder.trace[-1] == "[DONE]"
    assert len(decoder.trace) == 4


async def test_answer_stream_separates_reasoning_content() -> None:
    """A ``reasoning_content`` delta becomes a ``reasoning`` event."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(
            _sse(
                json.dumps({"choices": [{"delta": {"reasoning_content": "hmm "}}]}),
                json.dumps({"choices": [{"delta": {"content": "the answer"}}]}),
                json.dumps(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "m"}
                ),
                "[DONE]",
            )
        )

    events = [
        event async for event in _client(handler).answer_stream("q", _StreamDecoder())
    ]

    assert events == [
        Event("reasoning", "hmm "),
        Event("answer", "the answer"),
        Event("done", "m"),
    ]


async def test_answer_stream_reads_the_citations_extension() -> None:
    """A ``citations`` array on a chunk becomes a populated ``citations`` event."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(
            _sse(
                json.dumps(
                    {
                        "choices": [{"delta": {}}],
                        "citations": [
                            {
                                "position": 1,
                                "document_id": "d1",
                                "page_number": 4,
                                "section": "1 Intro",
                                "score": 0.7,
                            }
                        ],
                    }
                ),
                json.dumps({"choices": [{"delta": {"content": "a [1]"}}]}),
                json.dumps(
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "model": "m"}
                ),
                "[DONE]",
            )
        )

    events = [
        event async for event in _client(handler).answer_stream("q", _StreamDecoder())
    ]

    assert events[0].kind == "citations"
    citation = events[0].citations[0]
    assert (citation.document_id, citation.page_number, citation.section) == (
        "d1",
        4,
        "1 Intro",
    )
    assert citation.score == pytest.approx(0.7)


async def test_answer_stream_surfaces_a_mid_stream_error() -> None:
    """An ``error`` object mid-stream ends the stream as an ``error`` event."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _stream_response(
            _sse(
                json.dumps({"choices": [{"delta": {"content": "partial"}}]}),
                json.dumps({"error": {"message": "model exploded"}}),
                "[DONE]",
            )
        )

    events = [
        event async for event in _client(handler).answer_stream("q", _StreamDecoder())
    ]

    assert events == [
        Event("answer", "partial"),
        Event("error", "model exploded"),
    ]


async def test_answer_stream_raises_apierror_on_a_pre_stream_4xx() -> None:
    """A 4xx before the stream opens is surfaced as ``ApiError``."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    source = _client(handler).answer_stream("q", _StreamDecoder())
    with pytest.raises(ApiError) as excinfo:
        async for _ in source:
            pass

    assert excinfo.value.status_code == 429
    assert "rate limited" in str(excinfo.value)


async def test_empty_key_sends_no_authorization_header() -> None:
    """An empty key is not sent as a blank bearer header."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has_auth"] = "authorization" in request.headers
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m"}]})

    await _client(handler, api_key="").models()

    assert seen["has_auth"] is False


async def test_models_returns_the_first_model_and_adopts_it() -> None:
    """``GET /v1/models`` yields the first id and fills in the request model."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "gpt-x"}, {"id": "other"}]},
        )

    client = _client(handler)
    model = await client.models()

    assert seen["path"] == "/v1/models"
    assert model == "gpt-x"


async def test_versioned_base_url_is_not_doubled() -> None:
    """A base URL ending in ``/v1`` does not produce ``/v1/v1/...``."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"object": "list", "data": [{"id": "m"}]})

    client = OpenAIRagClient(
        "http://provider/v1", "k", transport=httpx.MockTransport(handler)
    )
    await client.models()

    assert seen["path"] == "/v1/models"


async def test_feedback_is_a_no_op_without_an_endpoint() -> None:
    """With no feedback URL, ``feedback`` reports "not recorded" and sends nothing."""
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(204)

    result = await _client(handler).feedback(
        vote="up",
        query="q",
        answer="a",
        model_name="m",
        citations=(),
    )

    assert result == "not recorded"
    assert called is False


async def test_feedback_posts_the_expected_shape_when_configured() -> None:
    """A configured endpoint receives the rating body, citations serialised."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(204)

    result = await _client(
        handler, feedback_url="http://provider/v1/feedback"
    ).feedback(
        vote="down",
        query="q",
        answer="a",
        model_name="m",
        citations=(Citation(1, "d1", 4, "1 Intro", 0.7),),
    )

    assert result == "recorded"
    assert captured["body"] == {
        "vote": "down",
        "query": "q",
        "answer": "a",
        "model_name": "m",
        "citations": [
            {
                "position": 1,
                "document_id": "d1",
                "page_number": 4,
                "section": "1 Intro",
                "score": 0.7,
            }
        ],
    }


async def test_ingest_posts_multipart_to_the_files_endpoint(tmp_path: object) -> None:
    """``ingest`` uploads the PDF to ``/v1/files`` and does nothing else."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"id": "file-1", "object": "file"})

    pdf = tmp_path / "doc.pdf"  # type: ignore[operator]
    pdf.write_bytes(b"%PDF-1.4 fake")

    outcome = await _client(handler).ingest(pdf)

    assert outcome["id"] == "file-1"
    assert calls == ["/v1/files"]


def test_decoder_state_does_not_leak_between_instances() -> None:
    """A fresh decoder starts with an empty trace and unset flags."""
    first = _StreamDecoder()
    list(
        first.decode(
            json.dumps(
                {"choices": [{"delta": {}}], "citations": [{"document_id": "d"}]}
            )
        )
    )
    list(first.decode("[DONE]"))

    second = _StreamDecoder()
    assert second.trace == []
    events = list(
        second.decode(json.dumps({"choices": [{"delta": {"content": "plain"}}]}))
    )
    assert events == [Event("answer", "plain")]
