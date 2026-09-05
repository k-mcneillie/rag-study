"""SQLAlchemy ORM models: the only place in the package that maps to tables.

These classes exist purely to persist and retrieve the domain contracts in
:mod:`rag.domain.models`. No processing component imports this module; the
pipelines work with domain objects and reach the database only through the
repository interface.

Two conventions in this schema come directly from the project's security
requirements. First, identifiers are opaque generated values, never anything
derived from a filename. Second, trusted provenance (identifiers, page numbers,
heading paths, model identity) is stored in typed columns, while untrusted
document-derived metadata is confined to the JSON ``metadata`` columns, so
document content cannot masquerade as control data.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DDL,
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    event,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from rag.storage.vector import Vector

#: Width of the ``VECTOR`` column, matching the default embedding model
#: (``all-MiniLM-L6-v2``, 384 dimensions). A model of a different width
#: requires a schema migration; the ``model_name`` and ``model_dimension``
#: columns on ``embeddings`` make any mismatch detectable rather than silent.
EMBEDDING_DIMENSION = 384

#: Length of the generated UUID identifiers used as primary keys.
_ID_LENGTH = 36

#: Table options applied to every table, pinning engine and collation.
_TABLE_ARGS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class DocumentRow(Base):
    """A row in the ``documents`` table."""

    __tablename__ = "documents"
    __table_args__ = _TABLE_ARGS

    id: Mapped[str] = mapped_column(String(_ID_LENGTH), primary_key=True)
    source_filename: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    # Mapped as ``extra_metadata`` because ``metadata`` is reserved by the
    # declarative base; the underlying column keeps the plain name.
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    pages: Mapped[list[PageRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class PageRow(Base):
    """A row in the ``pages`` table."""

    __tablename__ = "pages"
    __table_args__ = (
        Index("uq_pages_document_page", "document_id", "page_number", unique=True),
        _TABLE_ARGS,
    )

    id: Mapped[str] = mapped_column(String(_ID_LENGTH), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(_ID_LENGTH), ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    ocr_extracted: Mapped[bool] = mapped_column(Boolean, default=False)
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    document: Mapped[DocumentRow] = relationship(back_populates="pages")


class ChunkRow(Base):
    """A row in the ``chunks`` table."""

    __tablename__ = "chunks"
    __table_args__ = _TABLE_ARGS

    id: Mapped[str] = mapped_column(String(_ID_LENGTH), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(_ID_LENGTH), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    headers: Mapped[list] = mapped_column(JSON, default=list)
    text: Mapped[str] = mapped_column(Text)
    extra_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")
    embeddings: Mapped[list[EmbeddingRow]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )


class EmbeddingRow(Base):
    """A row in the ``embeddings`` table.

    A chunk may have more than one embedding when it has been re-embedded with
    a different model; ``model_name`` distinguishes them.
    """

    __tablename__ = "embeddings"
    __table_args__ = _TABLE_ARGS

    id: Mapped[str] = mapped_column(String(_ID_LENGTH), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        String(_ID_LENGTH), ForeignKey("chunks.id", ondelete="CASCADE"), index=True
    )
    # The column is named ``embedding`` because ``vector`` is a reserved word
    # in MariaDB 11.7 and later. NOT NULL is required for the vector index.
    vector: Mapped[tuple[float, ...]] = mapped_column(
        "embedding", Vector(EMBEDDING_DIMENSION), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), index=True)
    model_dimension: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chunk: Mapped[ChunkRow] = relationship(back_populates="embeddings")


# SQLAlchemy cannot express MariaDB's vector index inline, so it is added
# immediately after the table is created. The statement contains no external
# input.
event.listen(
    EmbeddingRow.__table__,
    "after_create",
    DDL(
        "ALTER TABLE embeddings "
        "ADD VECTOR INDEX ix_embeddings_embedding (embedding) DISTANCE=cosine"
    ),
)
