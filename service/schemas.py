"""The wire contract for the API.

These models are the boundary between the HTTP surface and the package's own
domain contracts. They exist so that a client — the Chainlit app in ``app/``,
or anything else — can be written against a small, documented shape without
importing ``rag``. Keeping them here, rather than reusing the frozen domain
dataclasses directly, means the HTTP shape can evolve without disturbing the
pipeline contracts, and vice versa.

The event grammar for ``POST /answer`` (a ``text/event-stream``) is not a
model here because it is a sequence of named events rather than one response
body; it is documented in ``docs/api.md`` and produced by
:mod:`service.streaming`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PromptId(BaseModel):
    """Identity of the prompt template that built a context block.

    Attributes:
        name: Logical template name, for example ``"retrieval_context"``.
        version: Template version, for example ``"v1"``.
    """

    name: str
    version: str


class CitationOut(BaseModel):
    """Provenance of one cited passage. Never carries the passage text.

    Attributes:
        position: One-based number of the passage in the prompt, matching the
            ``[n]`` marker the model was asked to cite.
        document_id: Identifier of the document the passage belongs to.
        page_number: Page the passage came from.
        section: Heading path within the document, if one was detected.
        score: The reranker's score for the passage.
    """

    position: int
    document_id: str
    page_number: int
    section: str | None = None
    score: float


class AnswerRequest(BaseModel):
    """A question to answer from the indexed corpus.

    Attributes:
        query: The question.
        top_k: How many passages to retrieve. ``None`` uses the service
            default; the value is clamped to the service maximum.
    """

    query: str
    top_k: int | None = Field(default=None, ge=1)


class HealthOut(BaseModel):
    """What the service is configured to do.

    Attributes:
        status: Always ``"ok"`` when the service is up.
        model: The answering model's identifier.
        provider: The configured generation provider name.
        reranker: ``"cross-encoder"`` when one is configured, else
            ``"vector-only"``.
        prompt: The prompt template in use, as ``"<name>/<version>"``.
        top_k: The default passage count.
    """

    status: Literal["ok"] = "ok"
    model: str
    provider: str
    reranker: Literal["cross-encoder", "vector-only"]
    prompt: str
    top_k: int


class FeedbackRequest(BaseModel):
    """One human rating of one answer.

    The client sends back everything the feedback log records. No passage text
    is included — only the provenance already kept in the log — so the corpus
    is not copied to a second place on disk.

    Attributes:
        vote: Whether the answer was judged helpful.
        query: The question that was asked.
        answer: The answer that was rated.
        model_name: Which model produced the answer.
        prompt: Identity of the prompt template used.
        citations: Provenance of the passages the answer was given.
    """

    vote: Literal["up", "down"]
    query: str
    answer: str
    model_name: str
    prompt: PromptId
    citations: list[CitationOut] = Field(default_factory=list)


class DocumentOut(BaseModel):
    """The outcome of an ingestion request.

    Attributes:
        status: ``"ingested"`` when the document was stored, or
            ``"already_indexed"`` when identical content was already present.
        document_id: Identifier assigned to a newly stored document; ``None``
            for a duplicate.
        chunk_count: How many chunks were stored, when the document was
            ingested.
    """

    status: Literal["ingested", "already_indexed"]
    document_id: str | None = None
    chunk_count: int | None = None
