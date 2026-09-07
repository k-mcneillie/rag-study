"""Reranking with a local cross-encoder.

The initial ranker compares a query vector against passage vectors that were
computed without ever seeing the query. That is what makes it cheap enough to
run over a whole corpus, and also what limits it: a passage about the right
subject scores well whether or not it answers the question, which is how an
"Author Contributions" section ends up alongside a paper's actual method.

A cross-encoder reads the query and one passage together and scores that pair
directly. It is far more accurate and far too slow to run over a store, so it
is applied to the shortlist the ranker produced — the two stages are a cheap
wide filter followed by an expensive narrow one.

Scores here are not comparable with the ranker's cosine similarities: a
cross-encoder emits an unbounded logit, not a similarity in [0, 1]. Both are
kept on the result — ``rerank_score`` and ``initial_score`` — so a change in
ordering can always be traced to which stage caused it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag.domain.models import RankedChunk, RerankedChunk
from rag.model_assets import ModelUnavailableError, load_cross_encoder
from rag.retrieval.interfaces import BaseReranker

__all__ = ["CrossEncoderReranker", "ModelUnavailableError"]


class CrossEncoderReranker(BaseReranker):
    """Reorders candidates by scoring each against the query directly.

    Attributes:
        model_path: Directory holding the cross-encoder's files.
        batch_size: How many query/passage pairs are scored at once.
        max_text_chars: Longest passage extract scored, bounding the cost of
            an unusually large chunk.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        batch_size: int = 16,
        max_text_chars: int = 2000,
    ) -> None:
        """Load the cross-encoder from a local directory.

        Args:
            model_path: Directory holding the model's files.
            batch_size: How many pairs to score at once, bounding peak memory.
            max_text_chars: Longest passage extract to score. A chunk longer
                than this is truncated for scoring only; the passage that
                reaches the prompt is never shortened.

        Raises:
            ModelUnavailableError: If the model cannot be loaded.
            ValueError: If the sizes are not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive.")

        self.model_path = model_path
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self._model = load_cross_encoder(model_path)

    def rerank(
        self, query: str, ranked_chunks: Sequence[RankedChunk], top_k: int
    ) -> list[RerankedChunk]:
        """Score every candidate against the query and reorder by that score.

        Args:
            query: The user's query.
            ranked_chunks: Candidates from the initial ranker.
            top_k: Maximum number of results to return.

        Returns:
            The candidates reordered by cross-encoder score, most relevant
            first, truncated to ``top_k``.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")
        if not ranked_chunks:
            return []

        pairs = [
            (query, candidate.chunk.text[: self.max_text_chars])
            for candidate in ranked_chunks
        ]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        ordered = sorted(
            zip(ranked_chunks, (float(score) for score in scores), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            RerankedChunk(ranked_chunk=candidate, rerank_score=score, rank=position)
            for position, (candidate, score) in enumerate(ordered[:top_k])
        ]
