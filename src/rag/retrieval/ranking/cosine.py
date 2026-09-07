"""Cosine-similarity ranking.

The comparison itself runs in the database, against the vector index, so only
the top-k rows travel back to the application rather than every embedding in
the store. That choice lives behind the repository, which returns scored
candidates without saying how it scored them, so moving the calculation into
Python later would change nothing here and nothing above.

Cosine similarity is confined to this class. Nothing else in the retrieval
pipeline assumes how relevance was computed: the orchestrator depends on
:class:`~rag.retrieval.interfaces.BaseRanker`, and a different measure — or a
hybrid of vector and keyword scoring — is a different implementation of the
same interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.domain.models import RankedChunk
from rag.retrieval.interfaces import BaseRanker
from rag.storage.repository import Repository


class CosineSimilarityRanker(BaseRanker):
    """Ranks chunks by cosine similarity to the query embedding."""

    def __init__(self, repository: Repository) -> None:
        """Initialise the ranker.

        Args:
            repository: The store to search. Supplied through the storage
                contract, so the ranker never touches a session or a table.
        """
        self._repository = repository

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
            filters: Optional scope restriction, validated by the repository
                against its allow-list.

        Returns:
            The candidates, ordered from most to least relevant.

        Raises:
            ValueError: If the repository rejects the request, for example
                because ``top_k`` is out of range or a filter is not allowed.
        """
        return self._repository.similarity_search(query_vector, top_k, filters=filters)
