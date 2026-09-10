"""An async HTTP client for an OpenAI-language RAG provider.

Two pieces:

* :class:`_StreamDecoder` turns one OpenAI ``data:`` payload into zero or more
  :class:`~rag_chat_openai.models.Event` values. The live stream and the offline
  demo both run through it, so the demo exercises the exact translation
  production uses and the rest of the app never sees the wire dialect. It also
  keeps every raw line in :attr:`_StreamDecoder.trace` for the "OpenAI wire
  trace" element.
* :class:`OpenAIRagClient` wraps an :class:`httpx.AsyncClient` with the
  ``Authorization: Bearer`` header set once and exposes the calls the app makes:
  the streamed answer, the model list, ingestion, and feedback.

The provider owns its index and retrieves internally; this client sends only the
question. The target is a *local* OpenAI-compatible server (vLLM, llama.cpp's
server, LM Studio, Ollama's ``/v1``, or a local RAG service speaking that
dialect), not the hosted OpenAI platform, so nothing here touches Assistants,
``/v1/vector_stores`` or hosted ``file_search``.

This module has no dependency on ``rag``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
from httpx_sse import aconnect_sse

from rag_chat_openai.models import Citation, Event

#: The sentinel that ends an OpenAI server-sent-event stream.
_DONE = "[DONE]"

#: Delta fields that carry reasoning when a provider separates it. Checked in
#: order; ``reasoning_content`` is the vLLM / DeepSeek spelling, ``reasoning``
#: the one some gateways use. A provider that instead inlines reasoning as
#: ``<think>`` tags in ``content`` is not handled — see ``docs/design.md``.
_REASONING_FIELDS = ("reasoning_content", "reasoning")

#: Path prefix for the versioned endpoints, dropped when the base URL already
#: ends in ``/v1`` so a request never goes to ``/v1/v1/...``.
_VERSION_SEGMENT = "/v1"


class ApiError(RuntimeError):
    """An error response from the provider.

    Attributes:
        status_code: The HTTP status returned.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        """Record the status and message.

        Args:
            status_code: The HTTP status returned.
            detail: The provider's error detail, or a fallback description.
        """
        super().__init__(detail)
        self.status_code = status_code


def _error_message(error: object) -> str:
    """Pull a human-readable message out of an OpenAI ``error`` value.

    Args:
        error: The value of a chunk's or body's ``error`` field, expected to be
            the ``{"message": ...}`` object the OpenAI dialect specifies.

    Returns:
        The ``message`` when present, else a generic fallback.
    """
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    return "The provider reported an error."


def _detail(response: httpx.Response) -> str:
    """Extract a message from an error response.

    Args:
        response: A response that has already been read.

    Returns:
        ``error.message`` when the body is the OpenAI error shape, else the body
        text, else the reason phrase.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(payload, dict) and "error" in payload:
        return _error_message(payload["error"])
    return response.text or response.reason_phrase


def _citations(items: list[dict[str, Any]]) -> tuple[Citation, ...]:
    """Build citations from a provider's ``citations`` extension array.

    Args:
        items: The array as the provider sent it. Each entry is expected to
            carry ``position``, ``document_id``, ``page_number``, ``section``
            and ``score``; missing values fall back rather than raise.

    Returns:
        The citations, in order.
    """
    built: list[Citation] = []
    for index, item in enumerate(items, start=1):
        built.append(
            Citation(
                position=int(item.get("position", index)),
                document_id=str(item.get("document_id", "")),
                page_number=int(item.get("page_number", 0)),
                section=item.get("section"),
                score=float(item.get("score", 0.0)),
            )
        )
    return tuple(built)


class _StreamDecoder:
    """Turns OpenAI ``data:`` payloads into :class:`Event` values.

    One instance per answer — the "citations already sent" and "stream already
    ended" flags must not carry across questions.

    Attributes:
        trace: Every raw ``data:`` payload seen, in order, including the final
            ``[DONE]``. Rendered by the "OpenAI wire trace" element.
    """

    def __init__(self) -> None:
        """Start with nothing decoded yet."""
        self._citations_sent = False
        self._done_sent = False
        self._last_model = ""
        self.trace: list[str] = []

    def decode(self, data: str) -> Iterator[Event]:
        """Decode one server-sent-event data payload.

        Args:
            data: The text after ``data:`` on one SSE line.

        Yields:
            The events the payload carries, if any.
        """
        self.trace.append(data)
        if data == _DONE:
            if not self._done_sent:
                self._done_sent = True
                yield Event("done", self._last_model or "unknown")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(chunk, dict):
            yield from self._decode_chunk(chunk)

    def _decode_chunk(self, chunk: dict[str, Any]) -> Iterator[Event]:
        """Decode one parsed ``chat.completion.chunk`` object.

        Args:
            chunk: The parsed JSON object.

        Yields:
            A ``citations`` event the first time the extension appears, then a
            ``reasoning`` and/or ``answer`` fragment for the delta, then a
            ``done`` on a terminal ``finish_reason``. An ``error`` object ends
            the stream immediately.
        """
        if (error := chunk.get("error")) is not None:
            self._done_sent = True
            yield Event("error", _error_message(error))
            return
        if model := chunk.get("model"):
            self._last_model = str(model)

        choice: dict[str, Any] = (chunk.get("choices") or [{}])[0]
        delta: dict[str, Any] = choice.get("delta") or {}

        if not self._citations_sent:
            raw = chunk.get("citations") or delta.get("citations")
            if isinstance(raw, list) and raw:
                self._citations_sent = True
                yield Event("citations", citations=_citations(raw))

        for field in _REASONING_FIELDS:
            if value := delta.get(field):
                yield Event("reasoning", str(value))
        if content := delta.get("content"):
            yield Event("answer", str(content))

        if choice.get("finish_reason") and not self._done_sent:
            self._done_sent = True
            yield Event("done", self._last_model or "unknown")


async def decode_lines(
    lines: AsyncIterator[str], decoder: _StreamDecoder
) -> AsyncIterator[Event]:
    """Run an async stream of raw ``data:`` payloads through a decoder.

    Used by the demo, which produces the payloads directly; the live client
    feeds ``decoder.decode`` inline instead.

    Args:
        lines: The raw payloads, one per SSE line.
        decoder: The decoder to feed. Its ``trace`` fills as a side effect.

    Yields:
        The decoded events.
    """
    async for line in lines:
        for event in decoder.decode(line):
            yield event


class OpenAIRagClient:
    """Talks to one OpenAI-language RAG provider."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str = "",
        feedback_url: str = "",
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Open a client against a provider.

        Args:
            base_url: Provider root, with or without a trailing ``/v1``.
            api_key: Bearer token. Omitted from requests when empty.
            model: Model id for the request body. Empty means it is filled in
                from :meth:`models` on first call.
            feedback_url: Where :meth:`feedback` posts. Empty disables storage.
            timeout: Read timeout in seconds. Generous, because generation is
                slow; the connect timeout stays short.
            transport: An alternative transport, used by the tests. Left unset
                in normal use.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )
        self._model = model
        self._feedback_url = feedback_url
        self._prefix = (
            "" if base_url.rstrip("/").endswith(_VERSION_SEGMENT) else _VERSION_SEGMENT
        )

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def models(self) -> str:
        """Fetch the provider's model list and adopt the first entry.

        The call doubles as the reachability probe that decides demo mode.

        Returns:
            The first model id the provider lists, or the configured model, or
            ``"unknown"``.

        Raises:
            httpx.HTTPError: If the provider is unreachable or returns an error
                status.
        """
        response = await self._client.get(f"{self._prefix}/models", timeout=10.0)
        response.raise_for_status()
        entries = response.json().get("data") or []
        first: dict[str, Any] = entries[0] if entries else {}
        model = str(first.get("id") or self._model or "unknown")
        if not self._model:
            self._model = model
        return model

    async def answer_stream(
        self, query: str, decoder: _StreamDecoder
    ) -> AsyncIterator[Event]:
        """Post the chat request and decode its SSE body.

        Args:
            query: The question to answer.
            decoder: The decoder to feed each ``data:`` line to. The caller
                reads ``decoder.trace`` once the stream is exhausted.

        Yields:
            The decoded events, in order.

        Raises:
            ApiError: If the provider rejects the request before the stream
                opens.
            httpx.HTTPError: If the connection fails mid-stream.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": query}],
            "stream": True,
        }
        async with aconnect_sse(
            self._client, "POST", f"{self._prefix}/chat/completions", json=payload
        ) as event_source:
            response = event_source.response
            if response.status_code >= 400:
                await response.aread()
                raise ApiError(response.status_code, _detail(response))
            async for sse in event_source.aiter_sse():
                for event in decoder.decode(sse.data):
                    yield event

    async def ingest(self, path: Path) -> dict[str, Any]:
        """Upload one PDF to the provider for indexing.

        Args:
            path: Path to a local PDF.

        Returns:
            The provider's JSON response for the created file.

        Raises:
            httpx.HTTPError: If the provider rejects the upload.
        """
        with path.open("rb") as handle:
            response = await self._client.post(
                f"{self._prefix}/files",
                files={"file": (path.name, handle, "application/pdf")},
                data={"purpose": "assistants"},
            )
        response.raise_for_status()
        outcome: dict[str, Any] = response.json()
        return outcome

    async def feedback(
        self,
        *,
        vote: str,
        query: str,
        answer: str,
        model_name: str,
        citations: Sequence[Citation],
    ) -> str:
        """Record one rating of an answer, if a feedback endpoint is configured.

        Args:
            vote: ``"up"`` or ``"down"``.
            query: The question that was asked.
            answer: The answer that was rated.
            model_name: The model that produced it.
            citations: Provenance of the passages the answer was given.

        Returns:
            ``"recorded"`` when posted, ``"not recorded"`` when no endpoint is
            configured.

        Raises:
            httpx.HTTPError: If the configured endpoint rejects the request.
        """
        if not self._feedback_url:
            return "not recorded"
        body: dict[str, Any] = {
            "vote": vote,
            "query": query,
            "answer": answer,
            "model_name": model_name,
            "citations": [dataclasses.asdict(item) for item in citations],
        }
        response = await self._client.post(self._feedback_url, json=body)
        response.raise_for_status()
        return "recorded"
