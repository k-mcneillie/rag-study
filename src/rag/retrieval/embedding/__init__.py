"""Query embedding, from local model assets only."""

from rag.retrieval.embedding.local import (
    LocalSentenceTransformerQueryEmbedder,
    ModelUnavailableError,
)

__all__ = ["LocalSentenceTransformerQueryEmbedder", "ModelUnavailableError"]
