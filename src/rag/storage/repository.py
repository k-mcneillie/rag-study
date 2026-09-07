"""The storage contract used by both pipelines.

This is the smallest interface that lets ingestion persist its output and lets
retrieval find candidates, and nothing more. It deliberately exposes no
sessions, tables, or query objects: a component that depends on this protocol
cannot reach the database directly, which is what allows the ranking and
prompting code to be tested against an in-memory implementation.

The protocol is the seam that keeps the two pipelines independent. Ingestion
uses only the ``save_*`` methods, retrieval uses only
:meth:`Repository.similarity_search`, and neither needs to know the other
exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from rag.domain.models import Document, EmbeddedChunk, Page, RankedChunk

#: Fields a caller is permitted to filter a similarity search on. Filtering is
#: restricted to an explicit allow-list so that a caller-supplied key can never
#: become an arbitrary column or SQL expression.
ALLOWED_SEARCH_FILTERS = frozenset({"document_id"})


@runtime_checkable
class Repository(Protocol):
    """Persistence operations required by the ingestion and retrieval pipelines."""

    def document_exists(self, content_hash: str) -> bool:
        """Report whether a document with this content has been stored.

        Lets ingestion skip a document it has already processed. Re-ingesting
        the same content would otherwise fill retrieval results with duplicate
        passages that crowd out other sources.

        Args:
            content_hash: Digest of the document's bytes.

        Returns:
            ``True`` if a document with that content hash is already stored.
        """
        ...

    def save_ingested_document(
        self,
        document: Document,
        pages: Sequence[Page],
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Persist a document, its pages, and its embedded chunks as one unit.

        This is deliberately a single operation rather than three. An ingested
        document is only useful complete: a document row without chunks is
        invisible to retrieval, yet its content hash makes every later attempt
        look like a duplicate to skip, so a partial write would remove the
        document from the store permanently and silently. Writing everything
        in one transaction means the store holds either the whole document or
        none of it.

        Re-ingesting the same document identifier replaces what is stored, so
        the operation can be repeated safely after a failure.

        Args:
            document: The document to store.
            pages: Its extracted pages.
            embedded_chunks: Its chunks, each with its embedding.
        """
        ...

    def similarity_search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[RankedChunk]:
        """Find the chunks most similar to a query embedding.

        Args:
            query_vector: The embedded query.
            top_k: Maximum number of results to return.
            filters: Optional scope restriction. Keys must come from
                :data:`ALLOWED_SEARCH_FILTERS`. This is the extension point
                for future access control; no access rules are applied today.

        Returns:
            The most similar chunks, ordered from most to least relevant, each
            carrying a relevance score where higher is more relevant.
        """
        ...
