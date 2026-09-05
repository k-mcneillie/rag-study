"""Tests for the OpenAI-compatible chat model.

This client is the evidence that the generation interface is provider-shaped
rather than Ollama-shaped: a different endpoint, different framing, different
reasoning conventions, and no change anywhere else in the system.

It is also the client vLLM is reached through. There is no vLLM server to test
against here, so what is tested is the protocol vLLM implements: the endpoint
path it serves, the server-sent event framing it uses, and the
``reasoning_content`` field it emits with a reasoning parser enabled.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import httpx
import pytest

from rag.config import GenerationSettings
from rag.domain.models import (
    Chunk,
    PromptContext,
    PromptMetadata,
    RankedChunk,
    RerankedChunk,
)
from rag.generation.interfaces import GenerationError
from rag.generation.openai_compatible import OpenAICompatibleChatModel

PROMPT = "Application rules.\n===== BEGIN SOURCE MATERIAL =====\nBody.\n"


def _context() -> PromptContext:
    """Build a prompt context without running retrieval.

    Returns:
        A context with one passage of provenance.
    """
    chunk = Chunk(document_id="doc-1", page_number=7, text="Body.")
    reranked = RerankedChunk(
        ranked_chunk=RankedChunk(chunk=chunk, score=0.8, rank=0),
        rerank_score=0.91,
        rank=0,
    )
    return PromptContext(
        chunks=(reranked,),
        rendered_text=PROMPT,
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )


def _events(*payloads: dict[str, object]) -> list[str]:
    """Frame payloads as the server-sent events the protocol uses.

    Args:
        payloads: The JSON objects to send, in order.

    Returns:
        The response's lines, terminated by the protocol's sentinel.
    """
    lines = [f"data: {json.dumps(payload)}" for payload in payloads]
    lines.append("data: [DONE]")
    return lines


def _delta(**fields: str) -> dict[str, object]:
    """Build one streamed choice delta.

    Args:
        fields: The delta's fields, for example ``content``.

    Returns:
        A payload in the protocol's shape.
    """
    return {"choices": [{"delta": fields}]}


def _stream_of(lines: Iterable[str]) -> httpx.MockTransport:
    """Build a transport that streams the given lines back.

    Args:
        lines: Raw lines to return.

    Returns:
        A mock transport serving that response to any request.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "".join(f"{line}\n" for line in lines)
        return httpx.Response(200, content=body.encode("utf-8"))

    return httpx.MockTransport(handler)


def _model(
    transport: httpx.MockTransport, **overrides: object
) -> OpenAICompatibleChatModel:
    """Build a client bound to a mock transport.

    Args:
        transport: The transport to use.
        overrides: Settings fields to override.

    Returns:
        The client under test.
    """
    fields: dict[str, object] = {
        "provider": "vllm",
        "base_url": "http://localhost:8000",
        "model": "test-model",
    }
    fields.update(overrides)
    settings = GenerationSettings(**fields)  # type: ignore[arg-type]
    return OpenAICompatibleChatModel(settings, transport=transport)


def test_the_rendered_prompt_is_sent_verbatim() -> None:
    """The prompt reaches the service exactly as the augmenter built it."""
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, content=b"data: [DONE]\n")

    list(_model(httpx.MockTransport(handler)).stream(_context()))

    assert sent["messages"] == [{"role": "user", "content": PROMPT}]
    assert sent["stream"] is True


def test_a_credential_is_sent_as_a_bearer_token() -> None:
    """A hosted service is reachable, which needs authentication."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=b"data: [DONE]\n")

    model = _model(httpx.MockTransport(handler), api_key="sk-test")
    list(model.stream(_context()))

    assert seen["authorization"] == "Bearer sk-test"


def test_no_authorization_header_without_a_credential() -> None:
    """A local server is not sent an empty credential."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, content=b"data: [DONE]\n")

    list(_model(httpx.MockTransport(handler)).stream(_context()))

    assert "authorization" not in seen


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000", "/v1/chat/completions"),
        ("http://localhost:8000/", "/v1/chat/completions"),
        ("http://localhost:8000/v1", "/v1/chat/completions"),
        ("http://localhost:8000/v1/", "/v1/chat/completions"),
        ("https://api.example.com/v1", "/v1/chat/completions"),
    ],
)
def test_the_version_segment_is_never_repeated(base_url: str, expected: str) -> None:
    """Both spellings of the base URL reach the same endpoint.

    vLLM and OpenAI are conventionally configured with ``/v1`` included and
    Ollama's compatible endpoint without it. Posting to ``/v1/v1/...`` returns
    a bare 404 that reads exactly like a missing model, so this is worth
    pinning down.

    Args:
        base_url: The configured base URL.
        expected: The path the request must be posted to.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, content=b"data: [DONE]\n")

    model = _model(httpx.MockTransport(handler), base_url=base_url)
    list(model.stream(_context()))

    assert seen == [expected]


def test_a_separate_reasoning_field_is_flagged_as_reasoning() -> None:
    """VLLM with a reasoning parser returns reasoning in its own field."""
    transport = _stream_of(
        _events(
            _delta(reasoning_content="Weighing "),
            _delta(reasoning_content="passage one."),
            _delta(content="The answer."),
        )
    )

    deltas = list(_model(transport).stream(_context()))

    assert [(d.text, d.reasoning) for d in deltas] == [
        ("Weighing ", True),
        ("passage one.", True),
        ("The answer.", False),
    ]


def test_inlined_think_tags_are_separated_from_the_answer() -> None:
    """A service without a reasoning field still yields a clean answer."""
    transport = _stream_of(
        _events(
            _delta(content="<think>Weighing"),
            _delta(content=" passage one.</think>The "),
            _delta(content="answer."),
        )
    )

    answer = _model(transport).answer(_context())

    assert answer.text == "The answer."
    assert answer.reasoning == "Weighing passage one."


def test_think_tags_split_across_fragments_are_handled() -> None:
    """The tag state carries between fragments, because tags straddle them."""
    transport = _stream_of(
        _events(
            _delta(content="Before <thi"),
            _delta(content="nk>hidden</thi"),
            _delta(content="nk> after."),
        )
    )

    answer = _model(transport).answer(_context())

    assert "hidden" not in answer.text


def test_state_does_not_leak_between_questions() -> None:
    """An unterminated reasoning block does not silence the next answer."""
    model = _model(_stream_of(_events(_delta(content="<think>never closed"))))
    list(model.stream(_context()))

    second = _model(_stream_of(_events(_delta(content="Plain answer."))))
    assert second.answer(_context()).text == "Plain answer."


def test_the_done_sentinel_yields_nothing() -> None:
    """The stream terminator is framing, not content."""
    transport = _stream_of(["data: [DONE]"])

    assert list(_model(transport).stream(_context())) == []


def test_non_data_lines_are_ignored() -> None:
    """Comment and keep-alive lines are not mistaken for malformed output."""
    transport = _stream_of([": keep-alive", "", *_events(_delta(content="Hi."))])

    assert [d.text for d in _model(transport).stream(_context())] == ["Hi."]


def test_a_rejected_credential_says_so() -> None:
    """A 401 names the credential rather than reporting a generic failure."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    with pytest.raises(GenerationError, match="RAG_LLM_API_KEY"):
        list(_model(httpx.MockTransport(handler)).stream(_context()))


def test_an_error_object_in_the_stream_is_raised() -> None:
    """An error mid-stream fails the call rather than being skipped."""
    transport = _stream_of(_events({"error": {"message": "context overflow"}}))

    with pytest.raises(GenerationError, match="context overflow"):
        list(_model(transport).stream(_context()))


def test_a_malformed_event_is_a_generation_error() -> None:
    """Unintelligible output fails loudly instead of yielding nothing."""
    transport = _stream_of(["data: {not json"])

    with pytest.raises(GenerationError, match="malformed response"):
        list(_model(transport).stream(_context()))


def test_text_held_back_as_a_possible_tag_is_still_emitted() -> None:
    """Withholding text is a delay, never a loss.

    A fragment ending in ``<`` might be the start of a tag, so it is held. If
    the stream then ends, it was only ever a less-than sign.
    """
    transport = _stream_of(_events(_delta(content="5 < 7 and 8 > 6")))

    assert _model(transport).answer(_context()).text == "5 < 7 and 8 > 6"


def test_an_unterminated_reasoning_block_stays_out_of_the_answer() -> None:
    """A truncated stream does not spill reasoning into the answer."""
    transport = _stream_of(_events(_delta(content="<think>cut off mid-thought")))

    answer = _model(transport).answer(_context())

    assert answer.text == ""
    assert answer.reasoning == "cut off mid-thought"
