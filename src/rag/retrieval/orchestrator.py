"""Coordination of the retrieval pipeline.

This module contains workflow and nothing else. It decides the order of the
stages and how many candidates each is given, but it holds no similarity
calculation, no database query, no reranking rule, and no prompt text. Every
stage arrives through the constructor, so replacing the ranker, the reranker,
or the prompt version needs no change here.

The orchestrator knows nothing about ingestion. It reads through the same
storage contract the ingestion pipeline writes to, which is the only thing the
two pipelines share.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from rag.domain.models import PromptContext
from rag.retrieval.interfaces import (
    BasePromptAugmenter,
    BaseQueryEmbedder,
    BaseRanker,
    BaseReranker,
)

logger = logging.getLogger(__name__)

#: How many candidates the ranker is asked for, relative to the number finally
#: returned. A reranker can only reorder what it is given, so it is given more
#: than will survive; without this it could only confirm the ranker's order.
_CANDIDATE_MULTIPLIER = 4


class RetrievalOrchestrator:
    """Runs a query through embedding, ranking, reranking, and prompting."""

    def __init__(
        self,
        *,
        query_embedder: BaseQueryEmbedder,
        ranker: BaseRanker,
        reranker: BaseReranker,
        prompt_augmenter: BasePromptAugmenter,
        max_top_k: int = 100,
        candidate_multiplier: int = _CANDIDATE_MULTIPLIER,
    ) -> None:
        """Wire the pipeline together.

        Args:
            query_embedder: Turns the query into a vector.
            ranker: Produces the initial candidate ordering.
            reranker: Refines that ordering.
            prompt_augmenter: Renders the final context.
            max_top_k: Largest result count a caller may request, bounding the
                cost of any single query.
            candidate_multiplier: How many candidates to retrieve per result
                returned, giving the reranker room to work.

        Raises:
            ValueError: If the limits are not positive.
        """
        if max_top_k < 1:
            raise ValueError("max_top_k must be at least 1.")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1.")

        self._query_embedder = query_embedder
        self._ranker = ranker
        self._reranker = reranker
        self._prompt_augmenter = prompt_augmenter
        self.max_top_k = max_top_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> PromptContext:
        """Retrieve context for a query.

        Args:
            query: The user's query.
            top_k: How many passages to return.
            filters: Optional scope restriction, validated by the repository
                against its allow-list.

        Returns:
            The assembled context, with each passage carrying its provenance
            and the whole carrying the identity of the prompt that built it.

        Raises:
            ValueError: If the query is empty or ``top_k`` is out of range.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1 or top_k > self.max_top_k:
            raise ValueError(
                f"top_k must be between 1 and {self.max_top_k}, got {top_k}."
            )

        query_vector = self._query_embedder.embed_query(query)
        candidate_count = min(top_k * self.candidate_multiplier, self.max_top_k)

        ranked = self._ranker.rank(query_vector, candidate_count, filters=filters)
        reranked = self._reranker.rerank(query, ranked, top_k)

        logger.info(
            "Retrieved %d candidates, returning %d passages.",
            len(ranked),
            len(reranked),
        )
        return self._prompt_augmenter.augment(query, reranked)
