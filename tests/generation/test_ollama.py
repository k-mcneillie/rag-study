"""Tests for the Ollama chat model.

Every test here runs against a mock transport. Nothing reaches the network,
and no server needs to be running: the client's job is to send an exact
request and to interpret a stream, and both are observable without one.
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
from rag.generation.ollama import OllamaChatModel

PROMPT = "Application rules.\n===== BEGIN SOURCE MATERIAL =====\nBody.\n"


def _context(rendered_text: str = PROMPT) -> PromptContext:
    """Build a prompt context without running retrieval.

    Args:
        rendered_text: The rendered prompt the context carries.

    Returns:
        A context with one passage of provenance.
    """
    chunk = Chunk(document_id="doc-1", page_number=7, text="Body.", section="2 Methods")
    reranked = RerankedChunk(
        ranked_chunk=RankedChunk(chunk=chunk, score=0.8, rank=0),
        rerank_score=0.91,
        rank=0,
    )
    return PromptContext(
        chunks=(reranked,),
        rendered_text=rendered_text,
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )


def _stream_of(lines: Iterable[str]) -> httpx.MockTransport:
    """Build a transport that streams the given lines back.

    Args:
        lines: Raw lines to return, newline-delimited.

    Returns:
        A mock transport serving that response to any request.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "".join(f"{line}\n" for line in lines)
        return httpx.Response(200, content=body.encode("utf-8"))

    return httpx.MockTransport(handler)


def _chat_lines(*, thinking: list[str], content: list[str]) -> list[str]:
    """Build a well-formed Ollama streaming response.

    Args:
        thinking: Reasoning fragments, emitted first.
        content: Answer fragments, emitted after the reasoning.

    Returns:
        The response's lines.
    """
    lines = [json.dumps({"message": {"thinking": part}}) for part in thinking]
    lines += [json.dumps({"message": {"content": part}}) for part in content]
    lines.append(json.dumps({"message": {"content": ""}, "done": True}))
    return lines


def _model(transport: httpx.MockTransport) -> OllamaChatModel:
    """Build a client bound to a mock transport.

    Args:
        transport: The transport to use.

    Returns:
        The client under test.
    """
    return OllamaChatModel(GenerationSettings(model="test-model"), transport=transport)


def test_the_rendered_prompt_is_sent_verbatim() -> None:
    """The prompt reaches the model exactly as the augmenter built it.

    The template's instructions and its source-material markers are one
    structure. Splitting or rewriting them here would move a boundary this
    component does not own.
    """
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, content=b'{"message":{"content":"ok"}}\n')

    model = _model(httpx.MockTransport(handler))
    list(model.stream(_context()))

    assert sent["messages"] == [{"role": "user", "content": PROMPT}]
    assert sent["model"] == "test-model"
    assert sent["stream"] is True


def test_sampling_settings_reach_the_service() -> None:
    """Configured sampling options are passed through, not defaulted away."""
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, content=b'{"message":{"content":"ok"}}\n')

    model = OllamaChatModel(
        GenerationSettings(temperature=0.7, num_ctx=4096, thinking=False),
        transport=httpx.MockTransport(handler),
    )
    list(model.stream(_context()))

    assert sent["think"] is False
    assert sent["options"] == {"temperature": 0.7, "num_ctx": 4096}


def test_reasoning_and_answer_are_kept_apart() -> None:
    """Thinking fragments are flagged, so a caller can present them apart."""
    transport = _stream_of(
        _chat_lines(thinking=["First, ", "consider."], content=["The ", "answer."])
    )

    deltas = list(_model(transport).stream(_context()))

    assert [(d.text, d.reasoning) for d in deltas] == [
        ("First, ", True),
        ("consider.", True),
        ("The ", False),
        ("answer.", False),
    ]


def test_the_completed_answer_carries_its_provenance() -> None:
    """A collected answer names its model and the prompt it answered."""
    transport = _stream_of(
        _chat_lines(thinking=["Hmm."], content=["Grounded ", "reply."])
    )

    answer = _model(transport).answer(_context())

    assert answer.text == "Grounded reply."
    assert answer.reasoning == "Hmm."
    assert answer.model_name == "test-model"
    assert answer.prompt_metadata.prompt_version == "v1"


def test_blank_lines_in_the_stream_are_ignored() -> None:
    """Keep-alive blank lines are not mistaken for malformed output."""
    transport = _stream_of(["", json.dumps({"message": {"content": "Hi."}}), ""])

    assert [d.text for d in _model(transport).stream(_context())] == ["Hi."]


def test_an_unreachable_service_is_reported_without_the_prompt() -> None:
    """A connection failure names the service, never the retrieved text."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    model = _model(httpx.MockTransport(handler))

    with pytest.raises(GenerationError) as caught:
        list(model.stream(_context()))

    assert "Cannot reach the model service" in str(caught.value)
    assert "SOURCE MATERIAL" not in str(caught.value)
    assert "Body." not in str(caught.value)


def test_a_missing_model_is_named_rather_than_pulled() -> None:
    """A 404 says which model to provision; nothing is downloaded."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    model = _model(httpx.MockTransport(handler))

    with pytest.raises(GenerationError, match="test-model is not available"):
        list(model.stream(_context()))


def test_a_rejected_request_reports_its_status() -> None:
    """Any other error status surfaces as a generation failure."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    model = _model(httpx.MockTransport(handler))

    with pytest.raises(GenerationError, match="HTTP 500"):
        list(model.stream(_context()))


def test_an_error_reported_mid_stream_is_raised() -> None:
    """An error object in the stream fails the call rather than being skipped."""
    transport = _stream_of([json.dumps({"error": "context length exceeded"})])

    with pytest.raises(GenerationError, match="context length exceeded"):
        list(_model(transport).stream(_context()))


def test_a_malformed_line_is_a_generation_error() -> None:
    """Unintelligible output fails loudly instead of yielding nothing."""
    transport = _stream_of(["{not json"])

    with pytest.raises(GenerationError, match="malformed response"):
        list(_model(transport).stream(_context()))


def test_the_model_name_is_reported_from_configuration() -> None:
    """The client identifies the model it was configured with."""
    assert _model(_stream_of([])).model_name == "test-model"
