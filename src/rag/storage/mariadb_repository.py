"""MariaDB implementation of the storage contract.

Ranking is delegated to the database: ``VEC_DISTANCE_COSINE`` runs against the
vector index, so only the top-k rows travel back to the application. The
:class:`~rag.storage.repository.Repository` protocol hides this choice, so
moving the calculation into Python later would not touch any caller.

All values reach the database through bound parameters built by SQLAlchemy
expressions. No SQL is assembled by string formatting anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy import Select, delete, func, insert, literal, or_, select
from sqlalchemy.orm import Session, sessionmaker

from rag.domain.models import Chunk, Document, EmbeddedChunk, Page, RankedChunk
from rag.storage.orm import ChunkRow, DocumentRow, EmbeddingRow, PageRow
from rag.storage.repository import ALLOWED_SEARCH_FILTERS
from rag.storage.vector import Vector


class MariaDBRepository:
    """Stores and retrieves pipeline data in MariaDB via SQLAlchemy.

    Attributes:
        embedding_dimension: Expected width of every vector handled here.
        max_top_k: Hard ceiling on the size of a single result set.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_dimension: int,
        max_top_k: int = 100,
    ) -> None:
        """Initialise the repository.

        Args:
            session_factory: Factory producing sessions bound to the target
                database.
            embedding_dimension: Width of the ``VECTOR`` column, used to
                reject mismatched vectors before they reach the database.
            max_top_k: Largest ``top_k`` a caller may request, bounding the
                cost of any single query.
        """
        self._session_factory = session_factory
        self.embedding_dimension = embedding_dimension
        self.max_top_k = max_top_k

    def document_exists(self, content_hash: str) -> bool:
        """Report whether a document with this content has been stored.

        Args:
            content_hash: Digest of the document's bytes.

        Returns:
            ``True`` if a document with that content hash is already stored.
        """
        statement = select(DocumentRow.id).where(
            DocumentRow.content_hash == content_hash
        )
        with self._session_factory() as session:
            return session.execute(statement.limit(1)).first() is not None

    def save_ingested_document(
        self,
        document: Document,
        pages: Sequence[Page],
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> None:
        """Persist a document, its pages, and its chunks in one transaction.

        Any earlier copy of the document is deleted first, so re-ingestion
        replaces rather than accumulates and a retry after a failure starts
        from a clean state. Foreign keys cascade, so removing the document row
        removes its pages, chunks, and embeddings with it.

        Rows are inserted in bulk rather than one at a time: a document of a
        few hundred chunks would otherwise cost a database round trip per
        chunk, which dominates ingestion time.

        Args:
            document: The document to store.
            pages: Its extracted pages.
            embedded_chunks: Its chunks, each with its embedding.

        Raises:
            ValueError: If any vector's width differs from the configured
                embedding dimension. Checked before the transaction opens, so
                a bad batch never partially writes.
        """
        for embedded in embedded_chunks:
            self._validate_dimension(embedded.vector)

        with self._session_factory() as session, session.begin():
            _delete_existing(session, document)
            session.add(
                DocumentRow(
                    id=document.id,
                    source_filename=document.source_filename,
                    content_hash=document.content_hash,
                    extra_metadata=dict(document.metadata),
                    created_at=document.created_at,
                )
            )
            session.flush()

            if pages:
                session.execute(insert(PageRow), [_page_row(page) for page in pages])
            if embedded_chunks:
                session.execute(
                    insert(ChunkRow),
                    [_chunk_row(embedded.chunk) for embedded in embedded_chunks],
                )
                session.execute(
                    insert(EmbeddingRow),
                    [_embedding_row(embedded) for embedded in embedded_chunks],
                )

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
            filters: Optional scope restriction, limited to the keys in
                :data:`~rag.storage.repository.ALLOWED_SEARCH_FILTERS`.

        Returns:
            The most similar chunks, ordered from most to least relevant.

        Raises:
            ValueError: If ``top_k`` is out of range, the query vector has the
                wrong width, or a filter key is not on the allow-list.
        """
        self._validate_top_k(top_k)
        self._validate_dimension(query_vector)

        statement = self._build_search_statement(query_vector, top_k, filters)

        with self._session_factory() as session:
            rows = session.execute(statement).all()

        return [
            RankedChunk(
                chunk=self._to_domain_chunk(chunk_row),
                # VEC_DISTANCE_COSINE returns a distance where smaller is
                # closer; the domain contract expects higher to be better.
                score=1.0 - float(distance),
                rank=rank,
            )
            for rank, (chunk_row, distance) in enumerate(rows)
        ]

    def _build_search_statement(
        self,
        query_vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, str] | None,
    ) -> Select[tuple[ChunkRow, float]]:
        """Build the similarity search query.

        Args:
            query_vector: The embedded query.
            top_k: Maximum number of results to return.
            filters: Optional scope restriction.

        Returns:
            The SELECT statement to execute.

        Raises:
            ValueError: If a filter key is not on the allow-list.
        """
        query_literal = literal(tuple(query_vector), Vector(self.embedding_dimension))
        distance = func.VEC_DISTANCE_COSINE(EmbeddingRow.vector, query_literal)

        statement = (
            select(ChunkRow, distance.label("distance"))
            .join(EmbeddingRow, EmbeddingRow.chunk_id == ChunkRow.id)
            .order_by(distance)
            .limit(top_k)
        )

        for key, value in self._validated_filters(filters).items():
            # Only allow-listed keys reach here, and each maps to a known
            # mapped attribute rather than to caller-supplied SQL.
            statement = statement.where(getattr(ChunkRow, key) == value)

        return statement

    @staticmethod
    def _validated_filters(filters: Mapping[str, str] | None) -> Mapping[str, str]:
        """Check that every filter key is permitted.

        Args:
            filters: The caller-supplied filters, if any.

        Returns:
            The filters, unchanged, or an empty mapping.

        Raises:
            ValueError: If a key is not on the allow-list.
        """
        if not filters:
            return {}
        rejected = set(filters) - ALLOWED_SEARCH_FILTERS
        if rejected:
            raise ValueError(
                f"Unsupported search filter(s): {sorted(rejected)}. "
                f"Allowed: {sorted(ALLOWED_SEARCH_FILTERS)}."
            )
        return filters

    def _validate_top_k(self, top_k: int) -> None:
        """Check that a requested result count is within bounds.

        Args:
            top_k: The requested number of results.

        Raises:
            ValueError: If ``top_k`` is below one or above ``max_top_k``.
        """
        if top_k < 1 or top_k > self.max_top_k:
            raise ValueError(
                f"top_k must be between 1 and {self.max_top_k}, got {top_k}."
            )

    def _validate_dimension(self, vector: Sequence[float]) -> None:
        """Check that a vector matches the configured embedding dimension.

        Args:
            vector: The vector to check.

        Raises:
            ValueError: If the vector's width is wrong.
        """
        if len(vector) != self.embedding_dimension:
            raise ValueError(
                f"Expected a {self.embedding_dimension}-dimensional vector, "
                f"got {len(vector)} dimensions."
            )

    @staticmethod
    def _to_domain_chunk(row: ChunkRow) -> Chunk:
        """Convert a chunk row into its domain equivalent.

        Args:
            row: The ORM row to convert.

        Returns:
            The corresponding domain chunk.
        """
        return Chunk(
            id=row.id,
            document_id=row.document_id,
            page_number=row.page_number,
            text=row.text,
            section=row.section,
            headers=tuple(row.headers or ()),
            ocr_extracted=row.ocr_extracted,
            metadata=dict(row.extra_metadata or {}),
        )


def _delete_existing(session: Session, document: Document) -> None:
    """Remove any stored copy of a document, and everything belonging to it.

    The identifiers are resolved with a query first and then deleted by
    explicit list, rather than by nesting subqueries inside the deletes.
    MariaDB restricts selecting from a table that the same statement deletes
    from, and resolving first sidesteps that entirely while making the
    intended set obvious.

    Children are removed innermost first rather than by relying on the foreign
    keys to cascade. The cascade is declared, but depending on it would make
    correctness here a property of the database's configuration rather than of
    this code.

    A document already stored under a different identifier but the same
    content is removed too, so re-ingesting a renamed file replaces it rather
    than storing it twice.

    Args:
        session: The open session, inside a transaction.
        document: The document being written.
    """
    document_ids = list(
        session.scalars(
            select(DocumentRow.id).where(
                or_(
                    DocumentRow.id == document.id,
                    DocumentRow.content_hash == document.content_hash,
                )
            )
        )
    )
    if not document_ids:
        return

    chunk_ids = list(
        session.scalars(
            select(ChunkRow.id).where(ChunkRow.document_id.in_(document_ids))
        )
    )
    if chunk_ids:
        session.execute(
            delete(EmbeddingRow).where(EmbeddingRow.chunk_id.in_(chunk_ids))
        )
        session.execute(delete(ChunkRow).where(ChunkRow.id.in_(chunk_ids)))

    session.execute(delete(PageRow).where(PageRow.document_id.in_(document_ids)))
    session.execute(delete(DocumentRow).where(DocumentRow.id.in_(document_ids)))
    session.flush()


def _page_row(page: Page) -> dict[str, object]:
    """Build the row values for a page.

    Args:
        page: The page to store.

    Returns:
        Column values for a bulk insert.
    """
    return {
        "id": page.id,
        "document_id": page.document_id,
        "page_number": page.page_number,
        "text": page.text,
        "ocr_extracted": page.ocr_extracted,
        "extra_metadata": dict(page.metadata),
    }


def _chunk_row(chunk: Chunk) -> dict[str, object]:
    """Build the row values for a chunk.

    Args:
        chunk: The chunk to store.

    Returns:
        Column values for a bulk insert.
    """
    return {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number,
        "section": chunk.section,
        "headers": list(chunk.headers),
        "text": chunk.text,
        "ocr_extracted": chunk.ocr_extracted,
        "extra_metadata": dict(chunk.metadata),
    }


def _embedding_row(embedded: EmbeddedChunk) -> dict[str, object]:
    """Build the row values for an embedding.

    Args:
        embedded: The embedded chunk to store.

    Returns:
        Column values for a bulk insert.
    """
    return {
        "id": embedded.id,
        "chunk_id": embedded.chunk.id,
        "vector": embedded.vector,
        "model_name": embedded.model_name,
        "model_dimension": embedded.model_dimension,
    }
