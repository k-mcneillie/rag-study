"""Turning a synchronous answer stream into server-sent events.

:meth:`~rag.generation.interfaces.BaseChatModel.stream` is a blocking
generator, by deliberate design of the package. The API is asynchronous. This
module bridges the two the same way ``app/main.py`` already does: each fragment
is pulled in a worker thread, so the event loop stays free without pushing
``async`` into ``rag``.

The event grammar it emits, consumed by any SSE client:

``citations``   once, after retrieval and before generation — the provenance
                of every passage the model is about to be given.
``delta``       one per fragment: ``{"reasoning": bool, "text": str}``.
``done``        once, after the last fragment: the model name and prompt id.
``error``       instead of ``done`` when generation fails; the message is
                already scrubbed of prompt and answer text by
                :class:`~rag.generation.interfaces.GenerationError`.
``no_context``  emitted by the route, not here, when retrieval found nothing
                and the model was never called.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator

from rag.domain.models import AnswerDelta, PromptContext
from rag.generation import BaseChatModel, GenerationError

logger = logging.getLogger(__name__)


def _pull_next(deltas: Iterator[AnswerDelta]) -> AnswerDelta | None:
    """Take the next fragment from a synchronous stream, or ``None`` at its end.

    Runs in a worker thread so the blocking ``next`` call does not stall the
    event loop.

    Args:
        deltas: The model's answer stream.

    Returns:
        The next fragment, or ``None`` when the stream is exhausted.
    """
    try:
        return next(deltas)
    except StopIteration:
        return None


def _citations_data(context: PromptContext) -> str:
    """Render the ``citations`` event payload for a context.

    Args:
        context: The context the model is about to be given.

    Returns:
        A JSON object with an ``items`` array, one entry per passage, carrying
        provenance only and never the passage text.
    """
    items = [
        {
            "position": position,
            "document_id": chunk.chunk.document_id,
            "page_number": chunk.chunk.page_number,
            "section": chunk.chunk.section,
            "score": chunk.rerank_score,
        }
        for position, chunk in enumerate(context.chunks, start=1)
    ]
    return json.dumps({"items": items})


def no_context_event() -> dict[str, str]:
    """Build the single event sent when retrieval matched nothing.

    Returns:
        An SSE ``no_context`` event. The model is not called in this case, so
        no ``delta`` or ``done`` follows.
    """
    return {"event": "no_context", "data": "{}"}


async def answer_events(
    chat_model: BaseChatModel, context: PromptContext
) -> AsyncIterator[dict[str, str]]:
    """Stream one answer as server-sent events.

    Args:
        chat_model: The model to answer with.
        context: The assembled retrieval context. Its ``rendered_text`` is
            sent to the model unchanged.

    Yields:
        SSE event dictionaries: ``citations``, then zero or more ``delta``,
        then either ``done`` or — if generation fails — ``error``.
    """
    yield {"event": "citations", "data": _citations_data(context)}

    deltas = chat_model.stream(context)
    try:
        while (delta := await asyncio.to_thread(_pull_next, deltas)) is not None:
            yield {
                "event": "delta",
                "data": json.dumps({"reasoning": delta.reasoning, "text": delta.text}),
            }
    except GenerationError as exc:
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "model_name": chat_model.model_name,
                "prompt": {
                    "name": context.prompt_metadata.prompt_name,
                    "version": context.prompt_metadata.prompt_version,
                },
            }
        ),
    }
