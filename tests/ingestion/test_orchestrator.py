"""Tests for the ingestion orchestrator.

The orchestrator is exercised entirely through fakes: no PDF, no embedding
model, and no database. That it can be tested this way is the point of the
constructor injection, and the fakes here mirror the ones the retrieval
pipeline will need.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag.domain.models import Chunk, Document, EmbeddedChunk, ExtractedDocument, Page
from rag.ingestion.embedding.local import (
    LocalSentenceTransformerEmbedder,
    ModelUnavailableError,
)
from rag.ingestion.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseExtractor,
    ExtractionError,
)
from rag.ingestion.orchestrator import IngestionOrchestrator


class FakeExtractor(BaseExtractor):
    """An extractor returning canned pages, or failing on demand."""

    def __init__(self, *, pages: int = 2, fail: bool = False) -> None:
        """Initialise the fake.

        Args:
            pages: How many pages to produce.
            fail: Whether to raise instead of extracting.
        """
        self.pages = pages
        self.fail = fail

    def extract(self, source: Path) -> ExtractedDocument:
        """Return canned pages for the source.

        Args:
            source: Path to the document.

        Returns:
            The canned extraction result.

        Raises:
            ExtractionError: If the fake is configured to fail.
        """
        if self.fail:
            raise ExtractionError(f"{source.name} could not be parsed as a PDF.")
        document = Document(source_filename=source.name, content_hash="hash-1")
        pages = tuple(
            Page(document_id=document.id, page_number=n, text=f"Page {n} body.")
            for n in range(1, self.pages + 1)
        )
        return ExtractedDocument(document=document, pages=pages)


class FakeChunker(BaseChunker):
    """A chunker producing one chunk per page, or none at all."""

    def __init__(self, *, empty: bool = False) -> None:
        """Initialise the fake.

        Args:
            empty: Whether to produce no chunks.
        """
        self.empty = empty

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Produce one chunk per page.

        Args:
            pages: The pages to chunk.

        Returns:
            The resulting chunks.
        """
        if self.empty:
            return []
        return [
            Chunk(
                document_id=page.document_id,
                page_number=page.page_number,
                text=page.text,
            )
            for page in pages
        ]


class FakeEmbedder(BaseEmbedder):
    """An embedder producing fixed vectors without loading a model."""

    @property
    def model_name(self) -> str:
        """Identity of the fake model.

        Returns:
            The model name.
        """
        return "fake-model"

    @property
    def dimension(self) -> int:
        """Width of the fake vectors.

        Returns:
            The dimensionality.
        """
        return 3

    def embed(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks with a fixed vector.

        Args:
            chunks: The chunks to embed.

        Returns:
            One embedded chunk per input chunk.
        """
        return [
            EmbeddedChunk(
                chunk=chunk, vector=(0.1, 0.2, 0.3), model_name=self.model_name
            )
            for chunk in chunks
        ]


class RecordingRepository:
    """A repository recording what it was asked to store."""

    def __init__(self, *, known_hashes: set[str] | None = None) -> None:
        """Initialise the fake.

        Args:
            known_hashes: Content hashes to report as already stored.
        """
        self.known_hashes = known_hashes or set()
        self.documents: list[Document] = []
        self.pages: list[Page] = []
        self.embedded: list[EmbeddedChunk] = []

    def document_exists(self, content_hash: str) -> bool:
        """Report whether the content is already stored.

        Args:
            content_hash: Digest to look up.

        Returns:
            Whether the hash is known.
        """
        return content_hash in self.known_hashes

    def save_document(self, document: Document) -> None:
        """Record a document.

        Args:
            document: The document to record.
        """
        self.documents.append(document)

    def save_pages(self, pages: Sequence[Page]) -> None:
        """Record pages.

        Args:
            pages: The pages to record.
        """
        self.pages.extend(pages)

    def save_chunks_with_embeddings(
        self, embedded_chunks: Sequence[EmbeddedChunk]
    ) -> None:
        """Record embedded chunks.

        Args:
            embedded_chunks: The embedded chunks to record.
        """
        self.embedded.extend(embedded_chunks)


def _orchestrator(
    *,
    extractor: BaseExtractor | None = None,
    chunker: BaseChunker | None = None,
    repository: RecordingRepository | None = None,
) -> tuple[IngestionOrchestrator, RecordingRepository]:
    """Assemble an orchestrator from fakes.

    Args:
        extractor: Extractor to use, defaulting to a working fake.
        chunker: Chunker to use, defaulting to a working fake.
        repository: Repository to use, defaulting to an empty recorder.

    Returns:
        The orchestrator and the repository it writes to.
    """
    store = repository or RecordingRepository()
    return (
        IngestionOrchestrator(
            extractor=extractor or FakeExtractor(),
            chunker=chunker or FakeChunker(),
            embedder=FakeEmbedder(),
            repository=store,
        ),
        store,
    )


def test_a_document_flows_through_every_stage() -> None:
    """Extraction, chunking, embedding, and storage all run in order."""
    orchestrator, store = _orchestrator()

    result = orchestrator.ingest(Path("paper.pdf"))

    assert result.succeeded
    assert result.chunk_count == 2
    assert len(store.documents) == 1
    assert len(store.pages) == 2
    assert len(store.embedded) == 2


def test_embeddings_carry_the_model_identity() -> None:
    """Every stored vector records which model produced it."""
    orchestrator, store = _orchestrator()

    orchestrator.ingest(Path("paper.pdf"))

    assert all(e.model_name == "fake-model" for e in store.embedded)


def test_an_unreadable_document_is_reported_not_raised() -> None:
    """One bad file yields a failed result rather than an exception."""
    orchestrator, store = _orchestrator(extractor=FakeExtractor(fail=True))

    result = orchestrator.ingest(Path("broken.pdf"))

    assert not result.succeeded
    assert result.error is not None
    assert store.documents == []


def test_a_batch_continues_past_a_failure() -> None:
    """A malformed document cannot end an ingestion run."""
    orchestrator, _ = _orchestrator(extractor=FakeExtractor(fail=True))

    results = orchestrator.ingest_all([Path("a.pdf"), Path("b.pdf")])

    assert len(results) == 2
    assert not any(result.succeeded for result in results)


def test_duplicate_documents_are_skipped() -> None:
    """Re-ingesting identical content does not duplicate it in the index."""
    store = RecordingRepository(known_hashes={"hash-1"})
    orchestrator, _ = _orchestrator(repository=store)

    result = orchestrator.ingest(Path("paper.pdf"))

    assert result.skipped
    assert not result.succeeded
    assert store.documents == []


def test_duplicate_checking_can_be_disabled() -> None:
    """Re-ingestion can be forced when a document must be reprocessed."""
    store = RecordingRepository(known_hashes={"hash-1"})
    orchestrator, _ = _orchestrator(repository=store)

    result = orchestrator.ingest(Path("paper.pdf"), skip_duplicates=False)

    assert result.succeeded
    assert len(store.documents) == 1


def test_a_document_yielding_no_text_is_reported() -> None:
    """A scan with no recoverable text fails clearly instead of storing nothing."""
    orchestrator, store = _orchestrator(chunker=FakeChunker(empty=True))

    result = orchestrator.ingest(Path("scan.pdf"))

    assert not result.succeeded
    assert result.error is not None
    assert store.documents == []


def test_a_missing_model_directory_fails_loudly(tmp_path: Path) -> None:
    """The embedder refuses to fall back to a download when weights are absent.

    Args:
        tmp_path: Temporary directory standing in for the model location.
    """
    with pytest.raises(ModelUnavailableError, match="never downloaded at runtime"):
        LocalSentenceTransformerEmbedder(tmp_path / "no-such-model")


def test_a_non_positive_batch_size_is_rejected(tmp_path: Path) -> None:
    """A nonsensical batch size fails before any model is loaded.

    Args:
        tmp_path: Temporary directory standing in for the model location.
    """
    with pytest.raises(ValueError, match="batch_size"):
        LocalSentenceTransformerEmbedder(tmp_path, batch_size=0)
