"""Tests that the storage contract is satisfiable without a database.

The in-memory implementation here is the practical proof that the
:class:`~rag.storage.repository.Repository` protocol leaks no database
concerns: it stores nothing but plain domain objects and ranks with arithmetic,
yet satisfies the same contract the MariaDB implementation does. Retrieval
components can therefore be tested with no server running.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytest

from rag.domain.models import Chunk, Document, EmbeddedChunk, Page, RankedChunk
from rag.storage.repository import ALLOWED_SEARCH_FILTERS, Repository


class InMemoryRepository:
    """A dependency-free repository backed by ordinary Python containers."""

    def __init__(self) -> None:
        """Initialise empty stores."""
        self.documents: list[Document] = []
        self.pages: list[Page] = []
        self.embedded_chunks: list[EmbeddedChunk] = []

    def document_exists(self, content_hash: str) -> bool:
        """Report whether a document with this content is already stored.

        Args:
            content_hash: Digest to look up.

        Returns:
            Whether a matching document is stored.
        """
        return any(document.content_hash == content_hash for document in self.documents)

    def save_ingested_document(
        self,
        document: Document,
        pages: Sequence[Page],
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Store a document, its pages, and its chunks together.

        Args:
            document: The document to store.
            pages: Its extracted pages.
            embedded_chunks: Its embedded chunks.
        """
        self.documents.append(document)
        self.pages.extend(pages)
        self.embedded_chunks.extend(embedded_chunks)

    def similarity_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedChunk]:
        """Rank stored chunks against a query embedding by cosine similarity.

        Args:
            query_vector: The embedded query.
            top_k: Maximum number of results to return.
            filters: Optional scope restriction.

        Returns:
            The most similar chunks, most relevant first.

        Raises:
            ValueError: If a filter key is not on the allow-list.
        """
        rejected = set(filters or {}) - ALLOWED_SEARCH_FILTERS
        if rejected:
            raise ValueError(f"Unsupported search filter(s): {sorted(rejected)}.")

        candidates = [
            embedded
            for embedded in self.embedded_chunks
            if all(
                getattr(embedded.chunk, key) == value
                for key, value in (filters or {}).items()
            )
        ]
        scored = sorted(
            ((self._cosine(query_vector, e.vector), e) for e in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            RankedChunk(chunk=embedded.chunk, score=score, rank=rank)
            for rank, (score, embedded) in enumerate(scored[:top_k])
        ]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            left: The first vector.
            right: The second vector.

        Returns:
            The cosine similarity, or 0.0 if either vector has zero length.
        """
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        magnitude = math.dist(left, [0.0] * len(left)) * math.dist(
            right, [0.0] * len(right)
        )
        return dot / magnitude if magnitude else 0.0


@pytest.fixture
def populated_repository() -> InMemoryRepository:
    """Provide a repository holding three chunks with known embeddings.

    Returns:
        A repository populated with orthogonal and near-duplicate vectors.
    """
    repository = InMemoryRepository()
    vectors = {"a": (1.0, 0.0, 0.0), "b": (0.9, 0.1, 0.0), "c": (0.0, 1.0, 0.0)}
    repository.save_ingested_document(
        Document(id="doc-1", source_filename="corpus.pdf", content_hash="hash-1"),
        [],
        [
            EmbeddedChunk(
                chunk=Chunk(
                    id=name, document_id=f"doc-{name}", page_number=1, text=name
                ),
                vector=vector,
                model_name="test-model",
            )
            for name, vector in vectors.items()
        ],
    )
    return repository


def test_in_memory_store_satisfies_the_repository_protocol() -> None:
    """The contract can be met without SQLAlchemy or a database."""
    assert isinstance(InMemoryRepository(), Repository)


def test_search_orders_results_by_relevance(
    populated_repository: InMemoryRepository,
) -> None:
    """Results come back most relevant first, with ranks assigned in order.

    Args:
        populated_repository: A repository with known embeddings.
    """
    results = populated_repository.similarity_search((1.0, 0.0, 0.0), top_k=3)

    assert [result.chunk.id for result in results] == ["a", "b", "c"]
    assert [result.rank for result in results] == [0, 1, 2]
    assert results[0].score > results[1].score > results[2].score


def test_search_respects_top_k(populated_repository: InMemoryRepository) -> None:
    """A request for fewer results returns only that many.

    Args:
        populated_repository: A repository with known embeddings.
    """
    assert len(populated_repository.similarity_search((1.0, 0.0, 0.0), top_k=2)) == 2


def test_search_can_be_scoped_to_a_document(
    populated_repository: InMemoryRepository,
) -> None:
    """The allow-listed filter restricts results to one document.

    Args:
        populated_repository: A repository with known embeddings.
    """
    results = populated_repository.similarity_search(
        (1.0, 0.0, 0.0), top_k=3, filters={"document_id": "doc-c"}
    )

    assert [result.chunk.id for result in results] == ["c"]


def test_unknown_filter_keys_are_rejected(
    populated_repository: InMemoryRepository,
) -> None:
    """A caller cannot filter on an arbitrary attribute.

    Args:
        populated_repository: A repository with known embeddings.
    """
    with pytest.raises(ValueError, match="Unsupported search filter"):
        populated_repository.similarity_search(
            (1.0, 0.0, 0.0), top_k=3, filters={"text": "'; DROP TABLE chunks; --"}
        )
