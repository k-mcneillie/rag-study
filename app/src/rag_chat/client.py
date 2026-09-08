"""An async HTTP client for the rag-study API.

One class, :class:`RagApiClient`, wrapping the four endpoints the app uses.
It holds an :class:`httpx.AsyncClient` with the ``X-API-Key`` header set once,
so every call — including the server-sent-events stream — carries it.

This module has no dependency on ``rag``; it speaks only the wire contract in
``models``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
from httpx_sse import aconnect_sse

from rag_chat.models import (
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


class ApiError(RuntimeError):
    """An error response from the API.

    Attributes:
        status_code: The HTTP status returned.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        """Record the status and message.

        Args:
            status_code: The HTTP status returned.
            detail: The service's error detail, or a fallback description.
        """
        super().__init__(detail)
        self.status_code = status_code


def _detail(response: httpx.Response) -> str:
    """Extract a human-readable message from an error response.

    Args:
        response: A response that has already been read.

    Returns:
        The ``detail`` field if the body is JSON, else the body text, else the
        reason phrase.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return response.text or response.reason_phrase


def _parse_event(event: str, data: str) -> StreamEvent:
    """Turn one server-sent event into a typed value.

    Args:
        event: The SSE event name.
        data: The SSE data line, a JSON object.

    Returns:
        The corresponding :data:`StreamEvent`.

    Raises:
        ValueError: If the event name is not one the contract defines.
    """
    payload = json.loads(data) if data else {}
    if event == "citations":
        return CitationsEvent(
            items=tuple(Citation(**item) for item in payload["items"])
        )
    if event == "delta":
        return DeltaEvent(text=payload["text"], reasoning=payload["reasoning"])
    if event == "done":
        return DoneEvent(
            model_name=payload["model_name"], prompt=PromptId(**payload["prompt"])
        )
    if event == "no_context":
        return NoContextEvent()
    if event == "error":
        return ErrorEvent(message=payload["message"])
    raise ValueError(f"Unknown stream event: {event!r}")


class RagApiClient:
    """Talks to one rag-study API service."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Open a client against a service.

        Args:
            base_url: Base URL of the service.
            api_key: Shared secret for the ``X-API-Key`` header. Omitted from
                requests when empty.
            timeout: Read timeout for a request, in seconds. Generation can be
                slow, so this is generous; the connect timeout is kept short so
                an unreachable service is detected quickly.
            transport: An alternative transport, used by the tests to stand in
                for a live service. Left unset in normal use.
        """
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()

    async def health(self) -> HealthInfo:
        """Fetch what the service is configured to do.

        Returns:
            The service's health report.

        Raises:
            httpx.HTTPError: If the service is unreachable or returns an error
                status.
        """
        response = await self._client.get("/health", timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return HealthInfo(
            model=data["model"],
            provider=data["provider"],
            reranker=data["reranker"],
            prompt=data["prompt"],
            top_k=data["top_k"],
        )

    async def answer_stream(
        self, query: str, *, top_k: int | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Stream an answer to a question.

        Args:
            query: The question to answer.
            top_k: How many passages to request, or ``None`` for the service
                default.

        Yields:
            The stream's events, in order.

        Raises:
            ApiError: If the service rejects the request (for example an empty
                query, 422).
            httpx.HTTPError: If the connection fails mid-stream.
        """
        payload: dict[str, object] = {"query": query}
        if top_k is not None:
            payload["top_k"] = top_k

        async with aconnect_sse(
            self._client, "POST", "/answer", json=payload
        ) as event_source:
            response = event_source.response
            if response.status_code >= 400:
                await response.aread()
                raise ApiError(response.status_code, _detail(response))
            async for sse in event_source.aiter_sse():
                yield _parse_event(sse.event, sse.data)

    async def feedback(
        self,
        *,
        vote: str,
        query: str,
        answer: str,
        model_name: str,
        prompt: PromptId,
        citations: Sequence[Citation],
    ) -> None:
        """Record one rating of an answer.

        Args:
            vote: ``"up"`` or ``"down"``.
            query: The question that was asked.
            answer: The answer that was rated.
            model_name: The model that produced it.
            prompt: The prompt template that built its context.
            citations: Provenance of the passages the answer was given.

        Raises:
            httpx.HTTPError: If the service rejects the request.
        """
        response = await self._client.post(
            "/feedback",
            json={
                "vote": vote,
                "query": query,
                "answer": answer,
                "model_name": model_name,
                "prompt": {"name": prompt.name, "version": prompt.version},
                "citations": [dataclasses.asdict(item) for item in citations],
            },
        )
        response.raise_for_status()

    async def ingest(self, path: Path) -> dict[str, object]:
        """Upload one PDF for indexing, blocking until it is done.

        Args:
            path: Path to a local PDF.

        Returns:
            The service's JSON response describing the outcome.

        Raises:
            httpx.HTTPError: If the service rejects the upload.
        """
        with path.open("rb") as handle:
            response = await self._client.post(
                "/documents",
                files={"file": (path.name, handle, "application/pdf")},
            )
        response.raise_for_status()
        outcome: dict[str, object] = response.json()
        return outcome
