"""The ingestion pipeline: documents in, embedded chunks stored.

Knows nothing about retrieval. It shares only the domain contracts and the
storage interface with the other pipeline.
"""

from rag.ingestion.chunking.pipeline import ChunkingPipeline
from rag.ingestion.chunking.semantic import SemanticChunker
from rag.ingestion.extraction.pdf import PDFExtractor
from rag.ingestion.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseExtractor,
    ExtractionError,
)
from rag.ingestion.orchestrator import IngestionOrchestrator, IngestionResult

__all__ = [
    "BaseChunker",
    "BaseEmbedder",
    "BaseExtractor",
    "ChunkingPipeline",
    "ExtractionError",
    "IngestionOrchestrator",
    "IngestionResult",
    "PDFExtractor",
    "SemanticChunker",
]
