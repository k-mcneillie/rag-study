"""The full cycle: a PDF in one end, retrieved context out the other.

Every other test exercises one stage. This one runs both pipelines against a
document generated for the purpose, and is the practical statement of the
architecture's central claim: ingestion writes through the storage contract,
retrieval reads through it, and neither knows the other exists.

It needs a live MariaDB and the local embedding model, and is skipped when
either is absent.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from sqlalchemy.orm import Session, sessionmaker

from rag.config import Settings
from rag.ingestion import ChunkingPipeline, IngestionOrchestrator, PDFExtractor
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder
from rag.retrieval import (
    CosineSimilarityRanker,
    PassthroughReranker,
    RetrievalOrchestrator,
    TemplatePromptAugmenter,
)
from rag.retrieval.embedding.local import LocalSentenceTransformerQueryEmbedder
from rag.storage.mariadb_repository import MariaDBRepository

pytestmark = pytest.mark.integration

#: A document with distinct sections, so retrieval has something to choose
#: between and the section trail has something to record.
PAGES = [
    (
        "Methods",
        "Participants arranged photographs on a touchscreen so that the "
        "distance between any two images reflected how similar they appeared. "
        "Each arrangement was recorded and converted into a dissimilarity "
        "matrix for later analysis by multidimensional scaling.",
    ),
    (
        "Results",
        "The scaling solution converged in three dimensions, and the recovered "
        "axes corresponded to shape, colour and typical size. Stress values "
        "fell below the conventional threshold for an acceptable fit.",
    ),
    (
        "Apparatus",
        "Stimuli were displayed on a calibrated monitor at a viewing distance "
        "of sixty centimetres, and responses were collected with a stylus.",
    ),
]


@pytest.fixture
def source_pdf(tmp_path: Path) -> Path:
    """Generate a small multi-section PDF.

    Args:
        tmp_path: Temporary directory for the document.

    Returns:
        Path to the generated PDF.
    """
    document = pymupdf.open()
    for heading, body in PAGES:
        page = document.new_page()
        page.insert_text((72, 72), heading, fontsize=18)
        page.insert_textbox(pymupdf.Rect(72, 100, 520, 400), body, fontsize=11)
    path = tmp_path / "study.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def repository(
    session_factory: sessionmaker[Session], integration_settings: Settings
) -> MariaDBRepository:
    """Provide a repository over the freshly created test schema.

    Args:
        session_factory: Factory for sessions on the test database.
        integration_settings: Settings supplying the embedding dimension.

    Returns:
        The repository both pipelines will use.
    """
    return MariaDBRepository(
        session_factory,
        embedding_dimension=integration_settings.embedding_dimension,
        max_top_k=integration_settings.max_top_k,
    )


@pytest.fixture
def ingested(
    source_pdf: Path,
    repository: MariaDBRepository,
    integration_settings: Settings,
) -> MariaDBRepository:
    """Run the ingestion pipeline over the generated document.

    Args:
        source_pdf: The document to ingest.
        repository: The store to write to.
        integration_settings: Settings supplying the model path.

    Returns:
        The repository, now populated.
    """
    if not integration_settings.embedding_model_path.is_dir():
        pytest.skip(f"No model at {integration_settings.embedding_model_path}")

    orchestrator = IngestionOrchestrator(
        extractor=PDFExtractor(use_ocr=False),
        chunker=ChunkingPipeline(chunk_size=600, chunk_overlap=80),
        embedder=LocalSentenceTransformerEmbedder(
            integration_settings.embedding_model_path
        ),
        repository=repository,
    )
    result = orchestrator.ingest(source_pdf)
    assert result.succeeded, result.error
    return repository


@pytest.fixture
def orchestrator(
    ingested: MariaDBRepository, integration_settings: Settings
) -> RetrievalOrchestrator:
    """Assemble retrieval over the populated store.

    Args:
        ingested: The populated repository.
        integration_settings: Settings supplying the model path.

    Returns:
        The retrieval orchestrator.
    """
    return RetrievalOrchestrator(
        query_embedder=LocalSentenceTransformerQueryEmbedder(
            integration_settings.embedding_model_path
        ),
        ranker=CosineSimilarityRanker(ingested),
        reranker=PassthroughReranker(),
        prompt_augmenter=TemplatePromptAugmenter(),
        max_top_k=integration_settings.max_top_k,
    )


def test_a_document_ingested_by_one_pipeline_is_found_by_the_other(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """The full cycle completes: PDF in, relevant context out.

    Args:
        orchestrator: Retrieval over the ingested document.
    """
    context = orchestrator.retrieve(
        "How did participants indicate how similar the images were?", top_k=2
    )

    assert context.chunks
    assert "touchscreen" in context.chunks[0].chunk.text


def test_retrieved_context_can_be_traced_back_to_the_page(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """Provenance survives the whole journey, not just one stage.

    Args:
        orchestrator: Retrieval over the ingested document.
    """
    context = orchestrator.retrieve("multidimensional scaling solution", top_k=1)

    retrieved = context.chunks[0].chunk
    assert retrieved.document_id
    assert retrieved.page_number in {1, 2, 3}
    assert retrieved.section
    assert f"page={retrieved.page_number}" in context.rendered_text


def test_the_prompt_records_which_template_built_it(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """A generated context is reproducible: it names its own prompt version.

    Args:
        orchestrator: Retrieval over the ingested document.
    """
    context = orchestrator.retrieve("stimuli display apparatus", top_k=1)

    assert context.prompt_metadata.prompt_name == "retrieval_context"
    assert context.prompt_metadata.prompt_version == "v1"


def test_re_ingesting_the_same_document_does_not_duplicate_it(
    source_pdf: Path,
    ingested: MariaDBRepository,
    integration_settings: Settings,
) -> None:
    """A second pass over the same file is skipped, keeping results clean.

    Args:
        source_pdf: The document already ingested.
        ingested: The populated repository.
        integration_settings: Settings supplying the model path.
    """
    orchestrator = IngestionOrchestrator(
        extractor=PDFExtractor(use_ocr=False),
        chunker=ChunkingPipeline(chunk_size=600, chunk_overlap=80),
        embedder=LocalSentenceTransformerEmbedder(
            integration_settings.embedding_model_path
        ),
        repository=ingested,
    )

    assert orchestrator.ingest(source_pdf).skipped
