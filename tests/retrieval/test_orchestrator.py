"""Tests for ranking, reranking, and the retrieval orchestrator.

The whole pipeline is exercised through fakes: no model, no database. That it
can be tested this way is the point of the constructor injection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from rag.domain.models import Chunk, RankedChunk, RerankedChunk
from rag.retrieval.interfaces import BaseQueryEmbedder, BaseRanker, BaseReranker
from rag.retrieval.orchestrator import RetrievalOrchestrator
from rag.retrieval.prompting.augmenter import TemplatePromptAugmenter
from rag.retrieval.ranking.cosine import CosineSimilarityRanker
from rag.retrieval.reranking.passthrough import PassthroughReranker


def _ranked(identifier: str, score: float, rank: int) -> RankedChunk:
    """Build a ranked chunk.

    Args:
        identifier: The chunk's identifier.
        score: Its relevance score.
        rank: Its position.

    Returns:
        The ranked chunk.
    """
    return RankedChunk(
        chunk=Chunk(
            id=identifier, document_id="doc-1", page_number=1, text=f"Text {identifier}"
        ),
        score=score,
        rank=rank,
    )


class FakeQueryEmbedder(BaseQueryEmbedder):
    """A query embedder returning a fixed vector."""

    def __init__(self) -> None:
        """Record the queries it is asked to embed."""
        self.queries: list[str] = []

    @property
    def dimension(self) -> int:
        """Width of the fake vectors.

        Returns:
            The dimensionality.
        """
        return 3

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Return a fixed vector, recording the query.

        Args:
            query: The query to embed.

        Returns:
            A fixed vector.
        """
        self.queries.append(query)
        return (1.0, 0.0, 0.0)


class RecordingRepository:
    """A repository recording the searches made against it."""

    def __init__(self, results: list[RankedChunk] | None = None) -> None:
        """Initialise the fake.

        Args:
            results: What to return from a search.
        """
        self.results = results if results is not None else []
        self.calls: list[tuple[int, Mapping[str, str] | None]] = []

    def document_exists(self, content_hash: str) -> bool:
        """Report that nothing is stored.

        Args:
            content_hash: Ignored.

        Returns:
            Always ``False``.
        """
        del content_hash
        return False

    def save_ingested_document(
        self,
        document: object,
        pages: Sequence[object],
        embedded_chunks: Sequence[object],
    ) -> None:
        """Ignore writes; retrieval never calls this.

        Args:
            document: Ignored.
            pages: Ignored.
            embedded_chunks: Ignored.
        """

    def similarity_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedChunk]:
        """Return canned results, recording how it was called.

        Args:
            query_vector: Ignored.
            top_k: Requested result count.
            filters: Requested scope restriction.

        Returns:
            The canned results, truncated to ``top_k``.
        """
        del query_vector
        self.calls.append((top_k, filters))
        return self.results[:top_k]


def _orchestrator(
    *,
    results: list[RankedChunk] | None = None,
    reranker: BaseReranker | None = None,
    ranker: BaseRanker | None = None,
) -> tuple[RetrievalOrchestrator, RecordingRepository]:
    """Assemble an orchestrator from fakes.

    Args:
        results: Canned search results.
        reranker: Reranker to use, defaulting to passthrough.
        ranker: Ranker to use, defaulting to cosine over the fake repository.

    Returns:
        The orchestrator and the repository beneath it.
    """
    repository = RecordingRepository(results)
    return (
        RetrievalOrchestrator(
            query_embedder=FakeQueryEmbedder(),
            ranker=ranker or CosineSimilarityRanker(repository),
            reranker=reranker or PassthroughReranker(),
            prompt_augmenter=TemplatePromptAugmenter(),
        ),
        repository,
    )


def test_ranker_delegates_to_the_repository() -> None:
    """The ranker asks the store rather than scoring in the application."""
    repository = RecordingRepository([_ranked("a", 0.9, 0)])

    results = CosineSimilarityRanker(repository).rank((1.0, 0.0, 0.0), 5)

    assert [r.chunk.id for r in results] == ["a"]
    assert repository.calls == [(5, None)]


def test_passthrough_reranker_preserves_order_and_scores() -> None:
    """The initial reranker changes nothing but the wrapper type."""
    ranked = [_ranked("a", 0.9, 0), _ranked("b", 0.5, 1)]

    reranked = PassthroughReranker().rerank("q", ranked, top_k=2)

    assert [r.chunk.id for r in reranked] == ["a", "b"]
    assert [r.rerank_score for r in reranked] == [0.9, 0.5]
    assert [r.initial_score for r in reranked] == [0.9, 0.5]


def test_passthrough_reranker_narrows_to_top_k() -> None:
    """Reranking returns only as many results as asked for."""
    ranked = [_ranked(str(i), 1.0 - i / 10, i) for i in range(10)]

    assert len(PassthroughReranker().rerank("q", ranked, top_k=3)) == 3


def test_reranker_rejects_a_nonsensical_top_k() -> None:
    """A zero or negative result count fails rather than returning nothing."""
    with pytest.raises(ValueError, match="top_k"):
        PassthroughReranker().rerank("q", [], top_k=0)


def test_retrieval_returns_context_with_provenance() -> None:
    """A query yields rendered context whose passages can be traced."""
    orchestrator, _ = _orchestrator(results=[_ranked("a", 0.9, 0)])

    context = orchestrator.retrieve("What was measured?", top_k=1)

    assert context.chunks[0].chunk.id == "a"
    assert "document=doc-1" in context.rendered_text
    assert context.prompt_metadata.prompt_version == "v1"


def test_the_reranker_is_given_more_candidates_than_are_returned() -> None:
    """A reranker that could only see the final results could not reorder."""
    orchestrator, repository = _orchestrator(
        results=[_ranked(str(i), 1.0 - i / 20, i) for i in range(40)]
    )

    orchestrator.retrieve("query", top_k=5)

    requested_top_k, _ = repository.calls[0]
    assert requested_top_k > 5


def test_candidate_retrieval_stays_within_the_hard_ceiling() -> None:
    """Widening the candidate pool cannot exceed the configured maximum."""
    repository = RecordingRepository([])
    orchestrator = RetrievalOrchestrator(
        query_embedder=FakeQueryEmbedder(),
        ranker=CosineSimilarityRanker(repository),
        reranker=PassthroughReranker(),
        prompt_augmenter=TemplatePromptAugmenter(),
        max_top_k=10,
        candidate_multiplier=100,
    )

    orchestrator.retrieve("query", top_k=10)

    assert repository.calls[0][0] == 10


def test_filters_are_passed_through_to_the_store() -> None:
    """A scope restriction reaches the repository that enforces it."""
    orchestrator, repository = _orchestrator(results=[])

    orchestrator.retrieve("query", top_k=1, filters={"document_id": "doc-9"})

    assert repository.calls[0][1] == {"document_id": "doc-9"}


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_rejected(query: str) -> None:
    """A blank query fails rather than embedding whitespace.

    Args:
        query: A blank query.
    """
    orchestrator, _ = _orchestrator()

    with pytest.raises(ValueError, match="must not be empty"):
        orchestrator.retrieve(query)


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_out_of_range_top_k_is_rejected(top_k: int) -> None:
    """Result counts are bounded, so one query cannot exhaust resources.

    Args:
        top_k: An out-of-range result count.
    """
    orchestrator, _ = _orchestrator()

    with pytest.raises(ValueError, match="top_k must be between"):
        orchestrator.retrieve("query", top_k=top_k)


@pytest.mark.parametrize(
    ("max_top_k", "multiplier"), [(0, 4), (-1, 4), (10, 0), (10, -1)]
)
def test_invalid_limits_are_rejected_at_construction(
    max_top_k: int, multiplier: int
) -> None:
    """Nonsensical limits fail when wiring, not on the first query.

    Args:
        max_top_k: The ceiling under test.
        multiplier: The candidate multiplier under test.
    """
    with pytest.raises(ValueError, match="at least 1"):
        RetrievalOrchestrator(
            query_embedder=FakeQueryEmbedder(),
            ranker=CosineSimilarityRanker(RecordingRepository()),
            reranker=PassthroughReranker(),
            prompt_augmenter=TemplatePromptAugmenter(),
            max_top_k=max_top_k,
            candidate_multiplier=multiplier,
        )


def test_no_results_still_produces_usable_context() -> None:
    """An unmatched query yields context that says so, not an error."""
    orchestrator, _ = _orchestrator(results=[])

    context = orchestrator.retrieve("something absent", top_k=3)

    assert context.chunks == ()
    assert "no relevant source material" in context.rendered_text


def test_a_custom_reranker_can_change_the_order() -> None:
    """The orchestrator honours whatever ordering the reranker returns."""

    class ReversingReranker(BaseReranker):
        """A reranker that reverses the ranking, to prove it is consulted."""

        def rerank(
            self, query: str, ranked_chunks: Sequence[RankedChunk], top_k: int
        ) -> list[RerankedChunk]:
            """Reverse the candidates.

            Args:
                query: Ignored.
                ranked_chunks: Candidates to reverse.
                top_k: Maximum results to return.

            Returns:
                The candidates in reverse order.
            """
            del query
            return [
                RerankedChunk(ranked_chunk=r, rerank_score=1.0 - i, rank=i)
                for i, r in enumerate(list(reversed(ranked_chunks))[:top_k])
            ]

    orchestrator, _ = _orchestrator(
        results=[_ranked("a", 0.9, 0), _ranked("b", 0.5, 1)],
        reranker=ReversingReranker(),
    )

    context = orchestrator.retrieve("query", top_k=2)

    assert [c.chunk.id for c in context.chunks] == ["b", "a"]
