"""Integration tests for the local embedding model.

These need the model weights present on disk and are skipped when they are
not, so the suite still runs on a machine that has not been provisioned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import ConfigurationError, load_settings
from rag.domain.models import Chunk
from rag.ingestion.embedding.local import LocalSentenceTransformerEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder() -> LocalSentenceTransformerEmbedder:
    """Load the configured local model, skipping if it is not provisioned.

    Returns:
        The embedder under test.
    """
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        pytest.skip(f"Configuration unavailable: {exc}")

    if not settings.embedding_model_path.is_dir():
        pytest.skip(f"No model at {settings.embedding_model_path}")

    return LocalSentenceTransformerEmbedder(
        settings.embedding_model_path, model_name=settings.embedding_model_name
    )


def _chunk(text: str) -> Chunk:
    """Build a chunk carrying the given text.

    Args:
        text: The chunk's text.

    Returns:
        The chunk.
    """
    return Chunk(document_id="doc-1", page_number=1, text=text)


def test_model_reports_its_identity_and_width(
    embedder: LocalSentenceTransformerEmbedder,
) -> None:
    """The embedder exposes the identity recorded with every vector.

    Args:
        embedder: The embedder under test.
    """
    assert embedder.model_name
    assert embedder.dimension == 384


def test_every_chunk_gets_a_vector_in_order(
    embedder: LocalSentenceTransformerEmbedder,
) -> None:
    """Embedding preserves both count and order.

    Args:
        embedder: The embedder under test.
    """
    chunks = [_chunk("First passage."), _chunk("Second passage."), _chunk("Third.")]

    embedded = embedder.embed(chunks)

    assert [e.chunk.text for e in embedded] == [c.text for c in chunks]
    assert all(e.model_dimension == embedder.dimension for e in embedded)


def test_vectors_match_the_schema_width(
    embedder: LocalSentenceTransformerEmbedder,
) -> None:
    """Vectors fit the VECTOR column without truncation.

    Args:
        embedder: The embedder under test.
    """
    assert len(embedder.embed([_chunk("A passage.")])[0].vector) == 384


def test_related_text_scores_above_unrelated_text(
    embedder: LocalSentenceTransformerEmbedder,
) -> None:
    """The model produces vectors that actually carry meaning.

    Args:
        embedder: The embedder under test.
    """
    query, related, unrelated = embedder.embed(
        [
            _chunk("How are language models aligned to human preferences?"),
            _chunk("Reinforcement learning from human feedback aligns models."),
            _chunk("The cake recipe calls for three eggs and butter."),
        ]
    )

    def similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        """Compute the dot product of two normalised vectors.

        Args:
            left: The first vector.
            right: The second vector.

        Returns:
            Their cosine similarity.
        """
        return sum(a * b for a, b in zip(left, right, strict=True))

    assert similarity(query.vector, related.vector) > similarity(
        query.vector, unrelated.vector
    )


def test_embedding_nothing_returns_nothing(
    embedder: LocalSentenceTransformerEmbedder,
) -> None:
    """An empty batch costs nothing and returns nothing.

    Args:
        embedder: The embedder under test.
    """
    assert embedder.embed([]) == []


def test_a_file_is_not_mistaken_for_a_model(tmp_path: Path) -> None:
    """A path that is not a model directory fails clearly.

    Args:
        tmp_path: Temporary directory for the fixture.
    """
    from rag.ingestion.embedding.local import ModelUnavailableError

    impostor = tmp_path / "model"
    impostor.write_text("not a model", encoding="utf-8")

    with pytest.raises(ModelUnavailableError, match="No model at"):
        LocalSentenceTransformerEmbedder(impostor)
