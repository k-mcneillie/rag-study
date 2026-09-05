"""Framework-independent domain contracts shared by both pipelines."""

from rag.domain.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    ExtractedDocument,
    Page,
    PromptContext,
    PromptMetadata,
    RankedChunk,
    RerankedChunk,
)

__all__ = [
    "Chunk",
    "Document",
    "EmbeddedChunk",
    "ExtractedDocument",
    "Page",
    "PromptContext",
    "PromptMetadata",
    "RankedChunk",
    "RerankedChunk",
]
