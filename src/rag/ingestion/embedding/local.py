"""Local, offline embedding with sentence-transformers.

The model is loaded from a directory on disk and nothing else. Every path that
would normally reach the Hugging Face Hub is closed: the directory is checked
before loading, the library is told to use local files only, and remote code
execution is refused. A missing or invalid model fails immediately with a
message naming the path, rather than silently reaching for the network and
succeeding on a machine that happens to be online.

Refusing remote code is a security boundary as much as an offline one. Loading
a model can execute code shipped alongside its weights, so a model directory is
trusted infrastructure, configured by an operator — never a path chosen by
a caller or derived from a document.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rag.domain.models import Chunk, EmbeddedChunk
from rag.ingestion.interfaces import BaseEmbedder

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _embedding_dimension(model: SentenceTransformer) -> int:
    """Read a model's output width across sentence-transformers versions.

    The accessor was renamed in recent releases; the old name still works but
    warns, so the new one is preferred where present.

    Args:
        model: The loaded model.

    Returns:
        The embedding dimensionality.
    """
    accessor = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    return int(accessor() or 0)


class ModelUnavailableError(RuntimeError):
    """Raised when a local model cannot be loaded.

    Names the configured path so the problem can be fixed, without implying
    that downloading the model is an option.
    """


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
            ModelUnavailableError: If the directory is missing or the model
                cannot be loaded from it.
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model_path = model_path
        self.batch_size = batch_size
        self._model_name = model_name or model_path.name
        self._model = self._load(model_path)
        self._dimension = _embedding_dimension(self._model)

    @staticmethod
    def _load(model_path: Path) -> SentenceTransformer:
        """Load the model, refusing any fallback to the network.

        Args:
            model_path: Directory holding the model's files.

        Returns:
            The loaded model.

        Raises:
            ModelUnavailableError: If the directory is missing or unloadable.
        """
        if not model_path.is_dir():
            raise ModelUnavailableError(
                f"No embedding model at {model_path}. Model weights are local "
                f"assets and are never downloaded at runtime; place the model "
                f"there and set RAG_EMBEDDING_MODEL_PATH accordingly."
            )

        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load embedding model from %s: %s",
                model_path,
                type(exc).__name__,
            )
            raise ModelUnavailableError(
                f"The embedding model at {model_path} could not be loaded."
            ) from exc

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
