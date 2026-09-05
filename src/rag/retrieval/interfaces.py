"""The replaceable stages of the retrieval pipeline.

Each interface answers the same four questions: what it receives, what it
returns, what it is responsible for, and what it knows nothing about.

:class:`BaseQueryEmbedder` is declared here rather than reused from the
ingestion package, even though both pipelines embed text with the same model.
Retrieval must not import ingestion, and the two need different things: an
ingestion embedder turns many chunks into embedded chunks in batches, while a
retrieval embedder turns one query string into one vector. Declaring the
narrower interface where it is used keeps the pipelines independent, and the
shared part — loading the model safely — lives in :mod:`rag.model_assets`,
which belongs to neither.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from rag.domain.models import PromptContext, RankedChunk, RerankedChunk


class BaseQueryEmbedder(ABC):
    """Turns a query into the vector used to search for candidates.

    Responsible for: loading a local model and encoding a single query.

    Knows nothing about: storage, ranking, reranking, or prompting.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Width of the vectors this embedder produces.

        Returns:
            The embedding dimensionality.
        """

    @abstractmethod
    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed a single query string.

        Args:
            query: The user's query.

        Returns:
            The query's embedding.
        """


class BaseRanker(ABC):
    """Produces an initial ordering of candidate chunks for a query.

    Responsible for: finding candidates and scoring their relevance.

    Knows nothing about: extraction, chunking, reranking, or prompting.
    """

    @abstractmethod
    def rank(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedChunk]:
        """Rank chunks against a query embedding.

        Args:
            query_vector: The embedded query.
            top_k: Maximum number of candidates to return.
            filters: Optional scope restriction.

        Returns:
            The candidates, ordered from most to least relevant.
        """


class BaseReranker(ABC):
    """Refines an existing ranking.

    Responsible for: reordering candidates and assigning rerank scores.

    Knows nothing about: extraction, chunking, storage, or prompting. A
    reranker may only reorder or narrow what the ranker returned; it never
    reaches back to the store for more.
    """

    @abstractmethod
    def rerank(
        self, query: str, ranked_chunks: Sequence[RankedChunk], top_k: int
    ) -> list[RerankedChunk]:
        """Reorder candidates for a query.

        Args:
            query: The user's query.
            ranked_chunks: Candidates from the initial ranker.
            top_k: Maximum number of results to return.

        Returns:
            The reranked results, most relevant first.
        """


class BasePromptAugmenter(ABC):
    """Assembles retrieved chunks into context for a downstream model.

    Responsible for: rendering a versioned template, keeping application
    instructions structurally separate from retrieved content, and recording
    which template produced the result.

    Knows nothing about: extraction, chunking, storage, or ranking — and it
    never calls a language model.
    """

    @abstractmethod
    def augment(self, query: str, chunks: Sequence[RerankedChunk]) -> PromptContext:
        """Build the context block for a query.

        Args:
            query: The user's query.
            chunks: The chunks to include, most relevant first.

        Returns:
            The rendered context together with its provenance.
        """
