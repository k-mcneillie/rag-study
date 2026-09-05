"""Coordination of the ingestion pipeline.

This module contains workflow and nothing else. It decides the order of the
stages, what happens when a document fails, and when a document can be skipped
— but it holds no extraction, chunking, or embedding logic of its own, and it
never mentions a specific model or file format. Every stage arrives through the
constructor, so swapping an extractor or embedder needs no change here.

The orchestrator knows nothing about retrieval. It writes through the same
storage contract the retrieval pipeline reads from, which is the only thing the
two pipelines share.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.ingestion.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseExtractor,
    ExtractionError,
)
from rag.storage.repository import Repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """What happened to one document.

    Attributes:
        source: The document's path.
        document_id: Identifier assigned to the document, or ``None`` if it
            was skipped or failed.
        chunk_count: How many chunks were stored.
        skipped: Whether the document was already present and left alone.
        error: A safe description of the failure, or ``None`` on success.
    """

    source: Path
    document_id: str | None = None
    chunk_count: int = 0
    skipped: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the document was ingested.

        Returns:
            ``True`` if the document was stored rather than skipped or failed.
        """
        return self.error is None and not self.skipped


class IngestionOrchestrator:
    """Runs documents through extraction, chunking, embedding, and storage."""

    def __init__(
        self,
        *,
        extractor: BaseExtractor,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        repository: Repository,
    ) -> None:
        """Wire the pipeline together.

        Args:
            extractor: Turns a source file into pages.
            chunker: Turns pages into chunks.
            embedder: Turns chunks into vectors.
            repository: Persists the result.
        """
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._repository = repository

    def ingest(self, source: Path, *, skip_duplicates: bool = True) -> IngestionResult:
        """Ingest a single document.

        Args:
            source: Path to the document.
            skip_duplicates: Whether to leave a document alone when one with
                identical content is already stored.

        Returns:
            What happened to the document. Failures are reported rather than
            raised, so that one unreadable file cannot end a batch.
        """
        try:
            extracted = self._extractor.extract(source)
        except ExtractionError as exc:
            logger.warning("Skipping %s: %s", source.name, exc)
            return IngestionResult(source=source, error=str(exc))

        document = extracted.document
        if skip_duplicates and self._repository.document_exists(document.content_hash):
            logger.info("Skipping %s: already ingested.", source.name)
            return IngestionResult(source=source, skipped=True)

        chunks = self._chunker.chunk(extracted.pages)
        if not chunks:
            logger.warning("Skipping %s: no text could be extracted.", source.name)
            return IngestionResult(
                source=source, error=f"{source.name} yielded no usable text."
            )

        embedded = self._embedder.embed(chunks)

        # The document and its pages are written before the chunks, so a chunk
        # never references a document that is not yet stored.
        self._repository.save_document(document)
        self._repository.save_pages(extracted.pages)
        self._repository.save_chunks_with_embeddings(embedded)

        logger.info("Ingested %s: %d chunks.", source.name, len(embedded))
        return IngestionResult(
            source=source, document_id=document.id, chunk_count=len(embedded)
        )

    def ingest_all(
        self, sources: Sequence[Path], *, skip_duplicates: bool = True
    ) -> list[IngestionResult]:
        """Ingest several documents, continuing past any that fail.

        Args:
            sources: Paths to the documents.
            skip_duplicates: Whether to skip documents already stored.

        Returns:
            One result per source, in the order given.
        """
        return [
            self.ingest(source, skip_duplicates=skip_duplicates) for source in sources
        ]
