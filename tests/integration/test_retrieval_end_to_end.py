"""End-to-end retrieval against a live MariaDB and the real local model.

This is the proof that the two pipelines meet only through storage. Nothing
here imports the ingestion package: the fixtures write rows through the same
repository the ingestion orchestrator uses, and retrieval finds them knowing
nothing about how they got there.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker

from rag.config import ConfigurationError, Settings, load_settings
from rag.domain.models import Chunk, Document, EmbeddedChunk
from rag.retrieval import (
    CosineSimilarityRanker,
    PassthroughReranker,
    RetrievalOrchestrator,
    TemplatePromptAugmenter,
)
from rag.retrieval.embedding.local import LocalSentenceTransformerQueryEmbedder
from rag.storage.mariadb_repository import MariaDBRepository

pytestmark = pytest.mark.integration

PASSAGES = {
    "dpo": (
        "Direct preference optimisation fits a policy to human preference "
        "pairs directly, without training a separate reward model first."
    ),
    "mds": (
        "Participants arranged images on a screen so that distance encoded "
        "perceived similarity, and the arrangements were scaled into a space."
    ),
    "pinn": (
        "Physics-informed neural networks constrain the loss with the "
        "residual of a partial differential equation at collocation points."
    ),
}


@pytest.fixture(scope="module")
def query_embedder() -> LocalSentenceTransformerQueryEmbedder:
    """Load the configured local model, skipping if it is not provisioned.

    Returns:
        The query embedder under test.
    """
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        pytest.skip(f"Configuration unavailable: {exc}")

    if not settings.embedding_model_path.is_dir():
        pytest.skip(f"No model at {settings.embedding_model_path}")

    return LocalSentenceTransformerQueryEmbedder(settings.embedding_model_path)


@pytest.fixture
def populated_repository(
    session_factory: sessionmaker[Session],
    integration_settings: Settings,
    query_embedder: LocalSentenceTransformerQueryEmbedder,
) -> MariaDBRepository:
    """Store three passages with real embeddings in the test schema.

    Args:
        session_factory: Factory for sessions on the test database.
        integration_settings: Settings supplying the embedding dimension.
        query_embedder: Used to embed the passages, standing in for ingestion.

    Returns:
        A repository holding the passages.
    """
    repository = MariaDBRepository(
        session_factory,
        embedding_dimension=integration_settings.embedding_dimension,
    )
    repository.save_document(
        Document(id="doc-1", source_filename="corpus.pdf", content_hash="hash-1")
    )
    repository.save_chunks_with_embeddings(
        [
            EmbeddedChunk(
                chunk=Chunk(
                    id=name,
                    document_id="doc-1",
                    page_number=index + 1,
                    text=text,
                    section=f"Section {name}",
                ),
                vector=query_embedder.embed_query(text),
                model_name="all-MiniLM-L6-v2",
            )
            for index, (name, text) in enumerate(PASSAGES.items())
        ]
    )
    return repository


@pytest.fixture
def orchestrator(
    populated_repository: MariaDBRepository,
    query_embedder: LocalSentenceTransformerQueryEmbedder,
) -> RetrievalOrchestrator:
    """Assemble the retrieval pipeline over the populated store.

    Args:
        populated_repository: The store holding the passages.
        query_embedder: Embeds the query.

    Returns:
        The orchestrator under test.
    """
    return RetrievalOrchestrator(
        query_embedder=query_embedder,
        ranker=CosineSimilarityRanker(populated_repository),
        reranker=PassthroughReranker(),
        prompt_augmenter=TemplatePromptAugmenter(),
    )


def test_a_query_finds_the_passage_that_answers_it(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """Vector search returns the relevant passage, not an arbitrary one.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    context = orchestrator.retrieve(
        "How can a policy be trained from preferences without a reward model?",
        top_k=1,
    )

    assert context.chunks[0].chunk.id == "dpo"


def test_a_different_query_finds_a_different_passage(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """Ranking discriminates between passages rather than always agreeing.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    context = orchestrator.retrieve(
        "How were image similarity judgements collected from participants?",
        top_k=1,
    )

    assert context.chunks[0].chunk.id == "mds"


def test_results_are_ordered_by_relevance(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """Scores decrease down the result list.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    context = orchestrator.retrieve("physics informed neural network", top_k=3)

    scores = [chunk.rerank_score for chunk in context.chunks]
    assert scores == sorted(scores, reverse=True)
    assert context.chunks[0].chunk.id == "pinn"


def test_retrieved_context_is_traceable_to_its_source(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """Every rendered passage carries the provenance needed to audit it.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    context = orchestrator.retrieve("preference optimisation", top_k=1)

    assert "document=doc-1" in context.rendered_text
    assert "page=" in context.rendered_text
    assert context.prompt_metadata.prompt_version == "v1"


def test_retrieval_can_be_scoped_to_a_document(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """The allow-listed filter reaches the database and restricts results.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    context = orchestrator.retrieve(
        "preference optimisation", top_k=3, filters={"document_id": "absent-doc"}
    )

    assert context.chunks == ()


def test_an_unsupported_filter_is_refused(
    orchestrator: RetrievalOrchestrator,
) -> None:
    """A caller cannot filter on an arbitrary column.

    Args:
        orchestrator: The retrieval pipeline under test.
    """
    with pytest.raises(ValueError, match="Unsupported search filter"):
        orchestrator.retrieve("query", top_k=1, filters={"text": "x"})
