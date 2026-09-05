"""The three replaceable stages of the ingestion pipeline.

Each stage answers the same four questions: what it receives, what it returns,
what it is responsible for, and what it knows nothing about. Keeping those
boundaries strict is what allows a new document type, chunking strategy, or
embedding model to be added without touching anything else.

None of these interfaces mention storage, ranking, or prompting. The pipeline
stages transform data; persisting the result is the orchestrator's business.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from rag.domain.models import Chunk, EmbeddedChunk, ExtractedDocument, Page


class ExtractionError(RuntimeError):
    """Raised when a source document cannot be extracted.

    Carries a message safe to surface to a caller: it names the document and
    the reason, without embedding filesystem internals or document content.
    """


class BaseExtractor(ABC):
    """Turns a source file into a document record and its pages.

    Responsible for: reading the source, preserving page boundaries, applying
    document-specific cleaning, and capturing metadata.

    Knows nothing about: chunking, embedding, ranking, storage, or prompting.
    """

    @abstractmethod
    def extract(self, source: Path) -> ExtractedDocument:
        """Extract a document's pages from a source file.

        Args:
            source: Path to the document to extract.

        Returns:
            The document record together with its extracted pages.

        Raises:
            ExtractionError: If the source is unreadable, unsupported, or
                exceeds a configured limit.
        """


class BaseChunker(ABC):
    """Turns extracted pages into retrievable chunks.

    Responsible for: understanding document structure and deciding chunk
    boundaries, sizes, and overlap.

    Knows nothing about: embeddings, storage, ranking, or prompting.
    """

    @abstractmethod
    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Split pages into chunks.

        Args:
            pages: The pages to split, in source order.

        Returns:
            The resulting chunks, in document order.
        """


class BaseEmbedder(ABC):
    """Turns chunks into vector representations.

    Responsible for: loading a local model and producing one vector per chunk,
    in batches.

    Knows nothing about: extraction, chunking strategy, storage, or ranking.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identity of the model, recorded alongside every stored vector.

        Returns:
            The model's name.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Width of the vectors this embedder produces.

        Returns:
            The embedding dimensionality.
        """

    @abstractmethod
    def embed(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed a sequence of chunks.

        Args:
            chunks: The chunks to embed.

        Returns:
            One embedded chunk per input chunk, in the same order.
        """
