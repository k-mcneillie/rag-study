"""A canned answer stream, so the UI runs with no service.

When ``GET /health`` cannot be reached — or when ``RAG_CHAT_DEMO`` is set —
the app streams these events instead of calling the API. They have the same
shape as a real ``POST /answer`` stream, so every part of the interface
(streaming, the reasoning step, the citations panel, the rating buttons) is
exercised. Nothing here is a real answer, and ratings are not recorded.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from rag_chat.models import (
    Citation,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    PromptId,
    StreamEvent,
)

_DEMO_CITATIONS = (
    Citation(
        position=1,
        document_id="demo-0000-0000-0000",
        page_number=3,
        section="2 Method > 2.1 Design",
        score=0.82,
    ),
    Citation(
        position=2,
        document_id="demo-0000-0000-0000",
        page_number=7,
        section="3 Results",
        score=0.61,
    ),
)


async def demo_events(query: str) -> AsyncIterator[StreamEvent]:
    """Yield a fixed answer stream for a question.

    Args:
        query: The question asked. Echoed into the canned answer so the demo
            visibly responds to input.

    Yields:
        The same event types a live ``POST /answer`` stream produces.
    """
    yield CitationsEvent(items=_DEMO_CITATIONS)

    for fragment in (
        "Demo mode: the API service is not reachable, ",
        "so nothing was retrieved and no model was called. ",
    ):
        yield DeltaEvent(text=fragment, reasoning=True)

    for fragment in (
        f"With a running service, an answer to “{query.strip()}” ",
        "would appear here, grounded in the indexed corpus and citing its ",
        "sources [1][2]. Start the service and reload to ask for real.",
    ):
        yield DeltaEvent(text=fragment, reasoning=False)

    yield DoneEvent(
        model_name="demo", prompt=PromptId(name="retrieval_context", version="v1")
    )
