"""The shapes the app works with internally.

These are the app's own types, not the provider's wire format. The client
(:mod:`rag_chat_openai.client`) decodes OpenAI ``chat.completion.chunk`` objects
*into* :class:`Event` values, and the demo (:mod:`rag_chat_openai.demo`)
produces the same chunks the decoder turns into them, so the rest of the app
never sees the wire dialect.

Both are frozen dataclasses with no dependency on ``rag`` or pydantic, so this
directory can be copied into another repository and run against a deployed
provider.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """Provenance of one cited passage.

    Attributes:
        position: One-based number matching the ``[n]`` marker in the answer.
        document_id: Identifier of the source document.
        page_number: Page the passage came from.
        section: Heading path within the document, if one was detected.
        score: The retriever's score for the passage.
    """

    position: int
    document_id: str
    page_number: int
    section: str | None
    score: float


@dataclass(frozen=True)
class Event:
    """One decoded step of an answer stream.

    A stream produces a ``citations`` event first (once, before any answer),
    then ``reasoning`` and ``answer`` fragments in the order the model emits
    them, then a single terminal ``done`` or ``error``.

    Attributes:
        kind: Which step this is — one of ``"citations"``, ``"reasoning"``,
            ``"answer"``, ``"done"`` or ``"error"``.
        text: The fragment for ``reasoning`` and ``answer``; the model name for
            ``done``; the failure message for ``error``; unused for
            ``citations``.
        citations: The cited passages, set only on a ``citations`` event.
    """

    kind: str
    text: str = ""
    citations: tuple[Citation, ...] = ()
