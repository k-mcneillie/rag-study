"""Local, offline embedding of queries.

The rules for loading a model safely live in :mod:`rag.model_assets`, shared
with the ingestion pipeline. A query must be embedded by the same model that
embedded the chunks, or the vectors occupy different spaces and similarity
between them is meaningless; the configured model path is what keeps them in
step, and the dimension check below turns a mismatch into an error rather than
silently poor results.
"""

from __future__ import annotations

from pathlib import Path

from rag.model_assets import (
    ModelUnavailableError,
    embedding_dimension,
    load_sentence_transformer,
)
from rag.retrieval.interfaces import BaseQueryEmbedder

__all__ = ["LocalSentenceTransformerQueryEmbedder", "ModelUnavailableError"]


class LocalSentenceTransformerQueryEmbedder(BaseQueryEmbedder):
    """Embeds queries using a sentence-transformers model held on disk.

    Attributes:
        model_path: Directory holding the model's files.
    """

    def __init__(self, model_path: Path) -> None:
        """Load the model from a local directory.

        Args:
            model_path: Directory holding the model's files.

        Raises:
            ModelUnavailableError: If the model cannot be loaded.
        """
        self.model_path = model_path
        self._model = load_sentence_transformer(model_path)
        self._dimension = embedding_dimension(self._model)

    @property
    def dimension(self) -> int:
        """Width of the vectors this embedder produces.

        Returns:
            The embedding dimensionality.
        """
        return self._dimension

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a single query string.

        Args:
            query: The user's query.

        Returns:
            The query's embedding.

        Raises:
            ValueError: If the query is empty or only whitespace.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return tuple(float(value) for value in vector)
