"""Ingest a directory of PDFs. The ingestion pipeline, end to end.

This script is assembly and nothing else: it reads configuration, constructs
each stage, and hands them to the orchestrator. All the behaviour lives in the
package, which is why swapping a component here is a one-line change.

Usage::

    python scripts/ingest.py path/to/documents

Nothing is downloaded. The embedding model must already be present at the
configured path, and the schema must already exist — see the README.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag.config import ConfigurationError, Settings, load_settings
from rag.ingestion import (
    ChunkingPipeline,
    IngestionOrchestrator,
    PDFExtractor,
    SemanticChunker,
)
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
from rag.model_assets import ModelUnavailableError
from rag.storage import MariaDBRepository, build_engine, build_session_factory


def build_orchestrator(settings: Settings) -> IngestionOrchestrator:
    """Assemble the ingestion pipeline from configuration.

    Args:
        settings: Application settings.

    Returns:
        The wired orchestrator.

    Raises:
        ModelUnavailableError: If the embedding model is not provisioned.
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


def main(argv: list[str] | None = None) -> int:
    """Ingest every PDF in a directory.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv``.

    Returns:
        A process exit status: ``0`` if every document was handled, ``1`` if
        configuration was missing or any document failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="Directory of PDFs to ingest.")
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Process documents even if identical content is already stored.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        settings = load_settings()
        orchestrator = build_orchestrator(settings)
    except (ConfigurationError, ModelUnavailableError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1

    sources = sorted(args.directory.glob("*.pdf"))
    if not sources:
        print(f"No PDFs found in {args.directory}.", file=sys.stderr)
        return 1

    results = orchestrator.ingest_all(sources, skip_duplicates=not args.reingest)

    ingested = [r for r in results if r.succeeded]
    skipped = [r for r in results if r.skipped]
    failed = [r for r in results if r.error]

    for result in failed:
        print(f"FAILED  {result.source.name}: {result.error}", file=sys.stderr)

    print(
        f"\nIngested {len(ingested)} documents "
        f"({sum(r.chunk_count for r in ingested)} chunks), "
        f"skipped {len(skipped)} duplicates, {len(failed)} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
