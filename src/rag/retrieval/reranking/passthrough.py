"""The initial reranker: keeps the ranker's ordering.

This is deliberately a no-op on the ordering. A reranker earns its cost when it
uses a stronger, more expensive model — typically a cross-encoder that reads
the query and a passage together rather than comparing two vectors — and no
such model is provisioned yet. Writing a heuristic in its place would mean
inventing a scoring rule with no evidence that it improves anything.

It is a real class rather than an absent stage so the pipeline is already
shaped for the reranker that will replace it. The orchestrator wires a
reranker in, results already carry rerank scores and their pre-rerank scores,
and swapping in a cross-encoder later changes one constructor argument.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.domain.models import RankedChunk, RerankedChunk
from rag.retrieval.interfaces import BaseReranker


class PassthroughReranker(BaseReranker):
    """Preserves the initial ranking, narrowing it to ``top_k``."""

    def rerank(
        self, query: str, ranked_chunks: Sequence[RankedChunk], top_k: int
    ) -> list[RerankedChunk]:
        """Keep the ranker's ordering and carry its scores forward.

        Args:
            query: The user's query, unused by this implementation.
            ranked_chunks: Candidates from the initial ranker.
            top_k: Maximum number of results to return.

        Returns:
            The candidates in their original order, as reranked results.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        del query
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        return [
            RerankedChunk(ranked_chunk=ranked, rerank_score=ranked.score, rank=position)
            for position, ranked in enumerate(ranked_chunks[:top_k])
        ]
