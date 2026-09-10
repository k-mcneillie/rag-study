"""A canned OpenAI-shaped chunk stream, so the UI runs with no provider.

When ``GET /v1/models`` cannot be reached — or when ``RAG_OPENAI_DEMO`` is set —
the app streams these raw ``data:`` payloads through the real
:class:`~rag_chat_openai.client._StreamDecoder`. So demo mode exercises the same
translation code a live answer does: the streaming, the reasoning step, the
citations panel, the rating buttons, and the "OpenAI wire trace" element are all
driven by exactly what production would receive. Nothing here is a real answer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

#: The two passages the canned answer "cites", as the provider's ``citations``
#: vendor extension would carry them.
_DEMO_CITATIONS: list[dict[str, object]] = [
    {
        "position": 1,
        "document_id": "demo-0000-0000-0000",
        "page_number": 3,
        "section": "2 Method > 2.1 Design",
        "score": 0.82,
    },
    {
        "position": 2,
        "document_id": "demo-0000-0000-0000",
        "page_number": 7,
        "section": "3 Results",
        "score": 0.61,
    },
]

_REASONING_FRAGMENTS = (
    "Demo mode: no provider is reachable, ",
    "so nothing was retrieved and no model was called. ",
)


def _chunk(
    delta: dict[str, object],
    *,
    citations: list[dict[str, object]] | None = None,
    finish_reason: str | None = None,
) -> str:
    """Render one ``chat.completion.chunk`` as its ``data:`` payload text.

    Args:
        delta: The ``choices[0].delta`` object for this chunk.
        citations: The ``citations`` vendor-extension array, on the one chunk
            that carries it (top level, as the provider contract puts it).
        finish_reason: The choice's ``finish_reason``, on the terminal chunk.

    Returns:
        A compact JSON string, exactly what a provider would put after
        ``data:``.
    """
    body: dict[str, object] = {
        "object": "chat.completion.chunk",
        "model": "demo",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if citations is not None:
        body["citations"] = citations
    return json.dumps(body)


def demo_chunks(query: str) -> tuple[str, ...]:
    """Build the fixed sequence of raw payloads for a question.

    Args:
        query: The question asked. Echoed into the canned answer so the demo
            visibly responds to input.

    Returns:
        The payloads in order: a citations chunk, two reasoning chunks, three
        answer chunks, a terminal chunk, then the literal ``[DONE]``.
    """
    answer_fragments = (
        f"With a provider wired up, an answer to “{query.strip()}” ",
        "would appear here, grounded in the indexed corpus and citing its ",
        "sources [1][2]. Set RAG_OPENAI_BASE_URL and reload to ask for real.",
    )
    payloads: list[str] = [
        _chunk({}, citations=_DEMO_CITATIONS),
        *(_chunk({"reasoning_content": fragment}) for fragment in _REASONING_FRAGMENTS),
        *(_chunk({"content": fragment}) for fragment in answer_fragments),
        _chunk({}, finish_reason="stop"),
        "[DONE]",
    ]
    return tuple(payloads)


async def demo_lines(query: str) -> AsyncIterator[str]:
    """Yield the canned payloads with a small delay, for a streaming feel.

    Args:
        query: The question asked.

    Yields:
        Each raw ``data:`` payload in turn.
    """
    for payload in demo_chunks(query):
        await asyncio.sleep(0.05)
        yield payload
