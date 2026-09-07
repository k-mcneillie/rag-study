"""Local, offline embedding of chunks.

The rules for loading a model safely live in :mod:`rag.model_assets`, which
both pipelines share. This module is only concerned with turning chunks into
embedded chunks in batches.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag.domain.models import Chunk, EmbeddedChunk
from rag.ingestion.interfaces import BaseEmbedder
from rag.model_assets import (
    ModelUnavailableError,
    embedding_dimension,
    load_sentence_transformer,
)

__all__ = ["LocalSentenceTransformerEmbedder", "ModelUnavailableError"]


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    """Embeds chunks using a sentence-transformers model held on disk.

    Attributes:
        model_path: Directory holding the model's files.
        batch_size: How many chunks are encoded per batch.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        model_name: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Load the model from a local directory.

        Args:
            model_path: Directory holding the model's files.
            model_name: Identity recorded alongside every vector. Defaults to
                the directory's name.
            batch_size: How many chunks to encode at a time, bounding peak
                memory for a large document.

        Raises:
            ModelUnavailableError: If the model cannot be loaded.
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model_path = model_path
        self.batch_size = batch_size
        self._model_name = model_name or model_path.name
        self._model = load_sentence_transformer(model_path)
        self._dimension = embedding_dimension(self._model)

    @property
    def model_name(self) -> str:
        """Identity of the model, recorded alongside every stored vector.

        Returns:
            The model's name.
        """
        return self._model_name

    @property
    def dimension(self) -> int:
        """Width of the vectors this embedder produces.

        Returns:
            The embedding dimensionality.
        """
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed plain strings.

        Satisfies the narrow embedding capability the semantic chunker
        depends on, so the same loaded model can serve both without the
        chunker importing anything from this package.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per input text, in the same order.
        """
        if not texts:
            return []

        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [tuple(float(value) for value in vector) for vector in vectors]

    def embed(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks in batches.

        Args:
            chunks: The chunks to embed.

        Returns:
            One embedded chunk per input chunk, in the same order.
        """
        if not chunks:
            return []

        vectors = self._model.encode(
            [chunk.text for chunk in chunks],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=tuple(float(value) for value in vector),
                model_name=self._model_name,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
