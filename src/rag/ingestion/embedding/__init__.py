"""Embedding: chunks in, vectors out, from local model assets only."""

from rag.ingestion.embedding.local import (
    LocalSentenceTransformerEmbedder,
    ModelUnavailableError,
)

__all__ = ["LocalSentenceTransformerEmbedder", "ModelUnavailableError"]
