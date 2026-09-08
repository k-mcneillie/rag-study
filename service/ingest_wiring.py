"""Assemble the ingestion pipeline for the ``POST /documents`` endpoint.

This duplicates ``scripts/ingest.py:build_orchestrator`` on purpose. The
package's :mod:`rag.assembly` composes the *query* side and is forbidden by
``tests/test_architecture.py`` from importing ``ingestion`` at all, so the
ingestion wiring cannot live there. Each entry point that ingests therefore
builds the pipeline itself, exactly as the scripts and the app each build the
query pipeline themselves. It is a few lines of construction and no behaviour.
"""

from __future__ import annotations

from rag.config import Settings
from rag.ingestion import (
    ChunkingPipeline,
    IngestionOrchestrator,
    PDFExtractor,
    SemanticChunker,
)
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
from rag.storage import MariaDBRepository, build_engine, build_session_factory


def build_ingestion_orchestrator(settings: Settings) -> IngestionOrchestrator:
    """Assemble the ingestion pipeline from configuration.

    Args:
        settings: Application settings.

    Returns:
        The wired orchestrator.

    Raises:
        ModelUnavailableError: If the embedding model is not provisioned. A
            configured model that cannot be loaded is an error, never a silent
            downgrade.
    """
    sessions = build_session_factory(build_engine(settings.database))
    embedder = LocalSentenceTransformerEmbedder(
        settings.embedding_model_path,
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
    )
    return IngestionOrchestrator(
        extractor=PDFExtractor(
            max_bytes=settings.max_document_bytes,
            max_pages=settings.max_document_pages,
        ),
        chunker=ChunkingPipeline(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            semantic_chunker=SemanticChunker(embedder)
            if settings.semantic_chunking
            else None,
        ),
        embedder=embedder,
        repository=MariaDBRepository(
            sessions,
            embedding_dimension=settings.embedding_dimension,
            max_top_k=settings.max_top_k,
        ),
    )
