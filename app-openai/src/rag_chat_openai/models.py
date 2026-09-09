"""The shapes the app works with internally.

These are the app's own event model, not the provider's wire format. The client
(:mod:`rag_chat_openai.client`) decodes OpenAI ``chat.completion.chunk`` objects
*into* these, and the demo (:mod:`rag_chat_openai.demo`) produces the same
chunks the decoder then turns into these, so the rest of the app never sees the
wire dialect. They are plain frozen dataclasses with no dependency on ``rag`` or
on pydantic, so this directory can be copied into another repository and run
against a deployed provider.

The ``StreamEvent`` union is the sequence one answer produces: ``CitationsEvent``
once, then ``DeltaEvent`` fragments, then ``DoneEvent`` — or ``NoContextEvent``
alone if a provider signals that retrieval matched nothing, or ``ErrorEvent``
when generation failed.
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
class PromptId:
    """Identity of the prompt template used.

    Attributes:
        name: Logical template name.
        version: Template version.
    """

    name: str
    version: str


@dataclass(frozen=True)
class HealthInfo:
    """What the provider reported at ``GET /v1/models``.

    Only ``model`` is guaranteed. The rest are populated when the provider
    decorates its model object with a ``rag`` block; otherwise they stay
    ``None`` and the banner simply omits them.

    Attributes:
        model: The answering model's identifier.
        provider: The configured generation provider, if reported.
        reranker: ``"cross-encoder"`` or ``"vector-only"``, if reported.
        prompt: The prompt template in use, as ``"<name>/<version>"``, if
            reported.
        top_k: The provider's default passage count, if reported.
    """

    model: str
    provider: str | None = None
    reranker: str | None = None
    prompt: str | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class CitationsEvent:
    """The passages the model was given.

    Attributes:
        items: The citations, in prompt order.
    """

    items: tuple[Citation, ...]


@dataclass(frozen=True)
class DeltaEvent:
    """One fragment of the streamed answer.

    Attributes:
        text: The fragment's text.
        reasoning: Whether the fragment is reasoning rather than answer.
    """

    text: str
    reasoning: bool


@dataclass(frozen=True)
class DoneEvent:
    """The end of a successful stream.

    Attributes:
        model_name: The model that answered.
        prompt: The prompt template that built the context, when the provider
            reports one. The OpenAI dialect has no field for it, so this is
            usually ``None``.
    """

    model_name: str
    prompt: PromptId | None


@dataclass(frozen=True)
class NoContextEvent:
    """Retrieval matched nothing; the model was not called."""


@dataclass(frozen=True)
class ErrorEvent:
    """Generation failed.

    Attributes:
        message: A description of the failure, provider-supplied.
    """

    message: str


StreamEvent = CitationsEvent | DeltaEvent | DoneEvent | NoContextEvent | ErrorEvent
