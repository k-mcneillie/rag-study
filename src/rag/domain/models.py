"""Domain contracts exchanged between pipeline components.

These models are the only vocabulary shared by the ingestion and retrieval
pipelines. They are deliberately free of any framework dependency: nothing in
this module imports SQLAlchemy, PyMuPDF, or any embedding library, so a change
of database or model provider cannot ripple into the data contract.

Every model is a frozen dataclass. Immutability is a security property, not a
stylistic choice: a chunk's provenance (its identifiers, page number, and
ranking scores) is set exactly once by the component responsible for it, and
document content can never overwrite it. Untrusted, document-derived values
live only in the ``metadata`` mappings, which no component is permitted to
interpret as instructions, identifiers, or control values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _new_id() -> str:
    """Generate an opaque internal identifier.

    Identifiers are generated rather than derived from user-supplied values
    such as filenames, so that they can never be used to escape a storage
    root or collide with an unrelated record.

    Returns:
        A randomly generated UUID4 string.
    """
    return str(uuid4())


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC timestamp.

    Returns:
        The current UTC time.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class Document:
    """A source document admitted to the ingestion pipeline.

    Attributes:
        source_filename: The original filename, retained for display and
            provenance only. It is untrusted input and must never be used as
            a filesystem path or identifier.
        content_hash: Hex digest of the document's bytes, used to detect
            duplicate or altered sources.
        id: Opaque internal identifier, generated rather than derived from
            the filename.
        metadata: Untrusted, document-derived metadata (for example the PDF
            author field). Never interpreted as instructions or control
            values.
        created_at: Timestamp at which the document was admitted.
    """

    source_filename: str
    content_hash: str
    id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class Page:
    """A single extracted page of a document.

    Pages preserve the document's original pagination so that retrieved
    context can always be traced back to a precise location in the source.

    Attributes:
        document_id: Identifier of the owning document.
        page_number: One-based page number within the source document.
        text: Extracted and cleaned page text.
        ocr_extracted: Whether the text came from OCR rather than from the
            document's own text layer.
        id: Opaque internal identifier.
        metadata: Untrusted, extractor-derived metadata.
    """

    document_id: str
    page_number: int
    text: str
    ocr_extracted: bool = False
    id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    """The complete result of extracting one source document.

    An extractor produces both the document record and its pages: it is the
    component that reads the file, so it is the only one positioned to derive
    the content hash and the document's own metadata.

    Attributes:
        document: The document record.
        pages: The extracted pages, in source order.
    """

    document: Document
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text produced by a chunker.

    Attributes:
        document_id: Identifier of the owning document.
        page_number: Page the chunk originated from.
        text: The chunk's text content. This is untrusted document content.
        section: Structural heading path for the chunk, if one was detected
            (for example ``"2 Methods > 2.1 Sampling"``).
        headers: The ordered heading trail leading to this chunk.
        id: Opaque internal identifier.
        metadata: Untrusted, chunker-derived metadata.
    """

    document_id: str
    page_number: int
    text: str
    section: str | None = None
    headers: tuple[str, ...] = ()
    id: str = field(default_factory=_new_id)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with the vector representation of its text.

    Attributes:
        chunk: The chunk that was embedded.
        vector: The embedding, as an immutable tuple of floats.
        model_name: Identity of the model that produced the vector, recorded
            so that a later change of embedding model is detectable rather
            than silent.
        id: Opaque internal identifier.
    """

    chunk: Chunk
    vector: tuple[float, ...]
    model_name: str
    id: str = field(default_factory=_new_id)

    @property
    def model_dimension(self) -> int:
        """Return the dimensionality of the embedding.

        Derived from the vector itself so that the recorded dimension and the
        stored vector can never disagree.

        Returns:
            The number of components in the embedding vector.
        """
        return len(self.vector)


@dataclass(frozen=True)
class RankedChunk:
    """A chunk with a relevance score assigned by a ranker.

    Attributes:
        chunk: The retrieved chunk.
        score: Relevance score, where higher is more relevant. Assigned only
            by the configured ranker; document content can never influence it
            directly.
        rank: Zero-based position in the ranked result set.
    """

    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class RerankedChunk:
    """A ranked chunk that has been rescored by a reranker.

    The original ranking is retained rather than discarded, so that the effect
    of the reranker on a given result remains auditable.

    Attributes:
        ranked_chunk: The result produced by the initial ranker.
        rerank_score: Score assigned by the reranker, where higher is more
            relevant.
        rank: Zero-based position in the reranked result set.
    """

    ranked_chunk: RankedChunk
    rerank_score: float
    rank: int

    @property
    def chunk(self) -> Chunk:
        """Return the underlying chunk.

        Returns:
            The chunk carried through from the initial ranking.
        """
        return self.ranked_chunk.chunk

    @property
    def initial_score(self) -> float:
        """Return the score assigned by the initial ranker.

        Returns:
            The pre-reranking relevance score.
        """
        return self.ranked_chunk.score


@dataclass(frozen=True)
class PromptMetadata:
    """Identity of the prompt template used to build a context block.

    Attributes:
        prompt_name: Logical name of the template, for example
            ``"retrieval_context"``.
        prompt_version: Version identifier of the template, for example
            ``"v1"``.
    """

    prompt_name: str
    prompt_version: str


@dataclass(frozen=True)
class PromptContext:
    """The final retrieval output: top-k context plus its provenance.

    This is the hand-off point to a future LLM. The retrieved text it carries
    is untrusted document content and must never be treated as instructions
    by whatever consumes it.

    Attributes:
        chunks: The reranked chunks included in the context, in order.
        rendered_text: The assembled context block, produced by applying the
            identified prompt template.
        prompt_metadata: Which prompt template produced ``rendered_text``.
    """

    chunks: tuple[RerankedChunk, ...]
    rendered_text: str
    prompt_metadata: PromptMetadata
