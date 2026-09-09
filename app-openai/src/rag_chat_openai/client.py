"""An async HTTP client for an OpenAI-language RAG provider.

Two pieces:

* :class:`_StreamDecoder` turns one OpenAI ``data:`` payload into zero or more
  :data:`~rag_chat_openai.models.StreamEvent` values. The live stream and the
  offline demo both run through it, so the demo exercises the exact translation
  production uses and the rest of the app never sees the wire dialect. It also
  keeps every raw line in :attr:`_StreamDecoder.trace` for the "OpenAI wire
  trace" element.
* :class:`OpenAIRagClient` wraps an :class:`httpx.AsyncClient` with the
  ``Authorization: Bearer`` header set once and exposes the calls the app makes:
  the streamed answer, the model list, ingestion, feedback, and an optional
  standalone search.

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

from rag_chat_openai.models import (
    Citation,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    HealthInfo,
    NoContextEvent,
    PromptId,
    StreamEvent,
)

#: The sentinel that ends an OpenAI server-sent-event stream.
_DONE = "[DONE]"

#: Delta fields that carry reasoning when a provider separates it. Checked in
#: order; ``reasoning_content`` is the vLLM / DeepSeek spelling, ``reasoning``
#: the one some gateways use.
_REASONING_FIELDS = ("reasoning_content", "reasoning")

#: Tags used by providers that inline reasoning in the answer instead.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

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
        error: The value of a chunk's or body's ``error`` field.

    Returns:
        The ``message`` when the error is an object, the string itself when it
        is a string, or a generic fallback.
    """
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    if isinstance(error, str) and error:
        return error
    return "The provider reported an error."


def _detail(response: httpx.Response) -> str:
    """Extract a message from an error response.

    Args:
        response: A response that has already been read.

    Returns:
        ``error.message`` or ``detail`` if the body is JSON, else the body
        text, else the reason phrase.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(payload, dict):
        if "error" in payload:
            return _error_message(payload["error"])
        if payload.get("detail"):
            return str(payload["detail"])
    return response.text or response.reason_phrase


def _partial_tag_length(text: str, tag: str) -> int:
    """Measure how much of ``text``'s tail could be the start of ``tag``.

    Args:
        text: The buffered text.
        tag: The tag being watched for.

    Returns:
        The length of the longest suffix of ``text`` that is a proper prefix of
        ``tag``, or zero when no suffix could become one.
    """
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


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
    """Turns OpenAI ``data:`` payloads into the app's ``StreamEvent`` union.

    One instance per answer — the ``<think>``-splitting state and the
    "citations already sent" flag must not carry across questions.

    Attributes:
        trace: Every raw ``data:`` payload seen, in order, including the final
            ``[DONE]``. Rendered by the "OpenAI wire trace" element.
    """

    def __init__(self) -> None:
        """Start with nothing decoded yet."""
        self._in_reasoning = False
        self._pending = ""
        self._citations_sent = False
        self._done_sent = False
        self._last_model = ""
        self.trace: list[str] = []

    def decode(self, data: str) -> Iterator[StreamEvent]:
        """Decode one server-sent-event data payload.

        Args:
            data: The text after ``data:`` on one SSE line.

        Yields:
            The events the payload carries, if any.
        """
        self.trace.append(data)
        if data == _DONE:
            yield from self._finish()
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return
        if isinstance(chunk, dict):
            yield from self._decode_chunk(chunk)

    def _finish(self) -> Iterator[StreamEvent]:
        """Flush held text and close the stream if nothing else did.

        Yields:
            A trailing ``DeltaEvent`` for any withheld text, then a synthesised
            ``DoneEvent`` when the provider ended on ``[DONE]`` without a
            ``finish_reason``.
        """
        yield from self._flush()
        if not self._done_sent:
            self._done_sent = True
            yield DoneEvent(model_name=self._last_model or "unknown", prompt=None)

    def _decode_chunk(self, chunk: dict[str, Any]) -> Iterator[StreamEvent]:
        """Decode one parsed ``chat.completion.chunk`` object.

        Args:
            chunk: The parsed JSON object.

        Yields:
            The events it carries.
        """
        if (error := chunk.get("error")) is not None:
            yield ErrorEvent(message=_error_message(error))
            self._done_sent = True
            return
        if model := chunk.get("model"):
            self._last_model = str(model)
        yield from self._maybe_citations(chunk)
        choice: dict[str, Any] = (chunk.get("choices") or [{}])[0]
        yield from self._deltas(choice.get("delta") or {})
        yield from self._maybe_end(chunk, choice)

    def _maybe_citations(self, chunk: dict[str, Any]) -> Iterator[StreamEvent]:
        """Emit the citations event the first time the extension appears.

        Args:
            chunk: The parsed chunk. The ``citations`` array may sit at the top
                level or inside ``choices[0].delta``.

        Yields:
            One ``CitationsEvent`` at most, ever.
        """
        if self._citations_sent:
            return
        raw = chunk.get("citations")
        if raw is None:
            choice: dict[str, Any] = (chunk.get("choices") or [{}])[0]
            raw = (choice.get("delta") or {}).get("citations")
        if isinstance(raw, list) and raw:
            self._citations_sent = True
            yield CitationsEvent(items=_citations(raw))

    def _deltas(self, delta: dict[str, Any]) -> Iterator[StreamEvent]:
        """Turn one ``delta`` object into answer and reasoning fragments.

        Args:
            delta: The ``choices[0].delta`` object.

        Yields:
            A reasoning ``DeltaEvent`` per reasoning field, then the answer
            content split at any ``<think>`` boundary.
        """
        for field in _REASONING_FIELDS:
            if value := delta.get(field):
                yield DeltaEvent(text=str(value), reasoning=True)
        if content := delta.get("content"):
            yield from self._split_inline_reasoning(str(content))

    def _maybe_end(
        self, chunk: dict[str, Any], choice: dict[str, Any]
    ) -> Iterator[StreamEvent]:
        """Close the stream on a ``finish_reason`` or a ``no_context`` signal.

        Args:
            chunk: The parsed chunk.
            choice: Its first ``choices`` entry.

        Yields:
            ``NoContextEvent`` when retrieval matched nothing, otherwise a
            ``DoneEvent`` once a terminal ``finish_reason`` arrives.
        """
        finish = choice.get("finish_reason")
        if finish == "no_context" or chunk.get("no_context"):
            yield from self._flush()
            self._done_sent = True
            yield NoContextEvent()
            return
        if finish and not self._done_sent:
            yield from self._flush()
            self._done_sent = True
            yield DoneEvent(model_name=self._last_model or "unknown", prompt=None)

    def _split_inline_reasoning(self, content: str) -> Iterator[StreamEvent]:
        """Separate inlined ``<think>`` reasoning from answer text.

        A tag can arrive split across fragments (``"<thi"`` then ``"nk>"``), so
        text that might still be the start of one is held back rather than
        emitted as answer.

        Args:
            content: One fragment of content text.

        Yields:
            The fragment, split at any tag boundary, minus any trailing text
            that could still become a tag.
        """
        self._pending += content
        while True:
            tag = _THINK_CLOSE if self._in_reasoning else _THINK_OPEN
            position = self._pending.find(tag)
            if position >= 0:
                if head := self._pending[:position]:
                    yield DeltaEvent(text=head, reasoning=self._in_reasoning)
                self._pending = self._pending[position + len(tag) :]
                self._in_reasoning = not self._in_reasoning
                continue
            held = _partial_tag_length(self._pending, tag)
            emit = self._pending[: len(self._pending) - held]
            self._pending = self._pending[len(self._pending) - held :]
            if emit:
                yield DeltaEvent(text=emit, reasoning=self._in_reasoning)
            return

    def _flush(self) -> Iterator[StreamEvent]:
        """Emit text held back in case it was the start of a tag.

        Yields:
            Whatever is still buffered. Text withheld as a possible tag was, in
            the end, just text.
        """
        if self._pending:
            yield DeltaEvent(text=self._pending, reasoning=self._in_reasoning)
            self._pending = ""


async def decode_lines(
    lines: AsyncIterator[str], decoder: _StreamDecoder
) -> AsyncIterator[StreamEvent]:
    """Run an async stream of raw ``data:`` payloads through a decoder.

    Used by the demo, which produces the payloads directly; the live client
    calls ``decoder.decode`` inline instead.

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
        vector_store: str = "",
        top_k: int = 5,
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
            vector_store: Optional index id, carried for :meth:`ingest` and
                :meth:`search` and available to :meth:`_retrieval_config`.
            top_k: Passage-count hint, available to :meth:`_retrieval_config`.
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
        self._vector_store = vector_store
        self._top_k = top_k
        self._feedback_url = feedback_url
        self._prefix = (
            "" if base_url.rstrip("/").endswith(_VERSION_SEGMENT) else _VERSION_SEGMENT
        )

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    def _retrieval_config(self) -> dict[str, Any]:
        """Provider-specific retrieval knobs for the chat request.

        Empty by default: a provider that owns its index retrieves without
        being told how. This is the single place to add ``tool_resources`` /
        ``vector_store_ids`` / a ``top_k`` hint once a concrete provider's
        contract is known — the values are already on the client
        (``self._vector_store``, ``self._top_k``).

        Returns:
            Extra keys to merge into the ``/v1/chat/completions`` body.
        """
        return {}

    async def models(self) -> HealthInfo:
        """Fetch the provider's model list and adopt the first model.

        Returns:
            A health report. Only ``model`` is guaranteed; the rest is filled
            from a ``rag`` block on the model object when the provider adds one.

        Raises:
            httpx.HTTPError: If the provider is unreachable or returns an error
                status.
        """
        response = await self._client.get(f"{self._prefix}/models", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        entries = data.get("data") or []
        first: dict[str, Any] = entries[0] if entries else {}
        model = str(first.get("id") or self._model or "unknown")
        extra: dict[str, Any] = first.get("rag") or {}
        if not self._model:
            self._model = model
        return HealthInfo(
            model=model,
            provider=extra.get("provider"),
            reranker=extra.get("reranker"),
            prompt=extra.get("prompt"),
            top_k=extra.get("top_k"),
        )

    def answer_stream(
        self, query: str
    ) -> tuple[AsyncIterator[StreamEvent], _StreamDecoder]:
        """Start a streamed answer.

        Args:
            query: The question to answer.

        Returns:
            The event stream and the decoder driving it. The caller reads
            ``decoder.trace`` once the stream is exhausted.
        """
        decoder = _StreamDecoder()
        return self._run_answer_stream(query, decoder), decoder

    async def _run_answer_stream(
        self, query: str, decoder: _StreamDecoder
    ) -> AsyncIterator[StreamEvent]:
        """Post the chat request and decode its SSE body.

        Args:
            query: The question to answer.
            decoder: The decoder to feed each ``data:`` line to.

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
            **self._retrieval_config(),
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
        if self._vector_store and outcome.get("id"):
            attach = await self._client.post(
                f"{self._prefix}/vector_stores/{self._vector_store}/files",
                json={"file_id": outcome["id"]},
            )
            attach.raise_for_status()
        return outcome

    async def feedback(
        self,
        *,
        vote: str,
        query: str,
        answer: str,
        model_name: str,
        prompt: PromptId | None,
        citations: Sequence[Citation],
    ) -> str:
        """Record one rating of an answer, if a feedback endpoint is configured.

        Args:
            vote: ``"up"`` or ``"down"``.
            query: The question that was asked.
            answer: The answer that was rated.
            model_name: The model that produced it.
            prompt: The prompt template that built its context, when known.
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
        if prompt is not None:
            body["prompt"] = {"name": prompt.name, "version": prompt.version}
        response = await self._client.post(self._feedback_url, json=body)
        response.raise_for_status()
        return "recorded"

    async def search(self, query: str) -> tuple[Citation, ...]:
        """Retrieve passages for a query without answering.

        Optional: used only for an explicit "show me the sources" action, never
        by the answer flow (the provider retrieves internally for that).

        Args:
            query: The search query.

        Returns:
            The ranked hits as citations, or an empty tuple when no vector
            store is configured.

        Raises:
            httpx.HTTPError: If the provider rejects the request.
        """
        if not self._vector_store:
            return ()
        response = await self._client.post(
            f"{self._prefix}/vector_stores/{self._vector_store}/search",
            json={"query": query},
        )
        response.raise_for_status()
        results = response.json().get("data") or []
        hits: list[Citation] = []
        for index, item in enumerate(results, start=1):
            attributes: dict[str, Any] = item.get("attributes") or {}
            hits.append(
                Citation(
                    position=index,
                    document_id=str(item.get("file_id") or item.get("filename") or ""),
                    page_number=int(attributes.get("page_number", 0)),
                    section=attributes.get("section"),
                    score=float(item.get("score", 0.0)),
                )
            )
        return tuple(hits)
