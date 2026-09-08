"""The shapes the app exchanges with the API.

These mirror the service's wire contract (``docs/api.md`` in the rag-study
repository) closely enough to work against, and no more. They are plain frozen
dataclasses with no dependency on ``rag`` or on pydantic, so this directory can
be copied into another repository and run against a deployed service.

The ``StreamEvent`` union is the sequence ``POST /answer`` streams:
``CitationsEvent`` once, then ``DeltaEvent`` fragments, then ``DoneEvent`` — or
``NoContextEvent`` alone when retrieval matched nothing, or ``ErrorEvent`` when
generation failed.
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
        score: The reranker's score for the passage.
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
    """What the service reported at ``GET /health``.

    Attributes:
        model: The answering model's identifier.
        provider: The configured generation provider.
        reranker: ``"cross-encoder"`` or ``"vector-only"``.
        prompt: The prompt template in use, as ``"<name>/<version>"``.
        top_k: The service's default passage count.
    """

    model: str
    provider: str
    reranker: str
    prompt: str
    top_k: int


@dataclass(frozen=True)
class CitationsEvent:
    """The passages the model is about to be given.

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
        prompt: The prompt template that built the context.
    """

    model_name: str
    prompt: PromptId


@dataclass(frozen=True)
class NoContextEvent:
    """Retrieval matched nothing; the model was not called."""


@dataclass(frozen=True)
class ErrorEvent:
    """Generation failed.

    Attributes:
        message: A description already scrubbed of prompt and answer text by
            the service.
    """

    message: str


StreamEvent = CitationsEvent | DeltaEvent | DoneEvent | NoContextEvent | ErrorEvent
