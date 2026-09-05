"""Integration tests for the MariaDB repository.

These run against a live MariaDB server using the disposable test schema, and
are skipped when no server is reachable. They cover the round trip through
SQLAlchemy and MariaDB's native ``VECTOR`` support, plus the security
behaviours the storage layer is responsible for.
"""

from __future__ import annotations

import pytest

from rag.domain.models import Chunk, Document, EmbeddedChunk, Page
from rag.storage.mariadb_repository import MariaDBRepository

pytestmark = pytest.mark.integration

DIMENSION = 384


def _vector(*leading: float) -> tuple[float, ...]:
    """Build a full-width vector from its leading components.

    Args:
        *leading: The first components; the remainder are zero.

    Returns:
        A vector of the schema's configured width.
    """
    return tuple(leading) + (0.0,) * (DIMENSION - len(leading))


def _embedded(
    chunk_id: str, document_id: str, text: str, vector: tuple[float, ...]
) -> EmbeddedChunk:
    """Build an embedded chunk for a test fixture.

    Args:
        chunk_id: Identifier for the chunk.
        document_id: Identifier of the owning document.
        text: The chunk's text.
        vector: The chunk's embedding.

    Returns:
        The assembled embedded chunk.
    """
    return EmbeddedChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=document_id,
            page_number=1,
            text=text,
            section="1 Introduction",
            headers=("1 Introduction",),
            metadata={"origin": "test"},
        ),
        vector=vector,
        model_name="test-model",
    )


def test_document_and_pages_round_trip(repository: MariaDBRepository) -> None:
    """Documents and their pages persist without error.

    Args:
        repository: Repository bound to the test schema.
    """
    document = Document(source_filename="paper.pdf", content_hash="deadbeef")
    repository.save_document(document)
    repository.save_pages(
        [
            Page(document_id=document.id, page_number=1, text="First page."),
            Page(
                document_id=document.id,
                page_number=2,
                text="Second page.",
                ocr_extracted=True,
            ),
        ]
    )


def test_similarity_search_ranks_by_cosine_distance(
    repository: MariaDBRepository,
) -> None:
    """MariaDB orders results by vector similarity to the query.

    Args:
        repository: Repository bound to the test schema.
    """
    document = Document(id="doc-1", source_filename="paper.pdf", content_hash="hash")
    repository.save_document(document)
    repository.save_chunks_with_embeddings(
        [
            _embedded("chunk-near", "doc-1", "closest", _vector(1.0, 0.0)),
            _embedded("chunk-mid", "doc-1", "close", _vector(0.9, 0.1)),
            _embedded("chunk-far", "doc-1", "orthogonal", _vector(0.0, 1.0)),
        ]
    )

    results = repository.similarity_search(_vector(1.0, 0.0), top_k=3)

    assert [result.chunk.id for result in results] == [
        "chunk-near",
        "chunk-mid",
        "chunk-far",
    ]
    assert [result.rank for result in results] == [0, 1, 2]
    assert results[0].score > results[2].score


def test_search_results_carry_full_provenance(repository: MariaDBRepository) -> None:
    """Retrieved chunks keep the provenance needed to audit a result.

    Args:
        repository: Repository bound to the test schema.
    """
    repository.save_document(
        Document(id="doc-1", source_filename="p.pdf", content_hash="h")
    )
    repository.save_chunks_with_embeddings(
        [_embedded("chunk-1", "doc-1", "body text", _vector(1.0))]
    )

    result = repository.similarity_search(_vector(1.0), top_k=1)[0]

    assert result.chunk.id == "chunk-1"
    assert result.chunk.document_id == "doc-1"
    assert result.chunk.page_number == 1
    assert result.chunk.section == "1 Introduction"
    assert result.chunk.headers == ("1 Introduction",)
    assert result.chunk.metadata == {"origin": "test"}


def test_search_can_be_scoped_to_one_document(repository: MariaDBRepository) -> None:
    """The allow-listed filter keeps results within a single document.

    Args:
        repository: Repository bound to the test schema.
    """
    repository.save_document(
        Document(id="doc-1", source_filename="a.pdf", content_hash="h1")
    )
    repository.save_document(
        Document(id="doc-2", source_filename="b.pdf", content_hash="h2")
    )
    repository.save_chunks_with_embeddings(
        [
            _embedded("chunk-1", "doc-1", "from document one", _vector(1.0)),
            _embedded("chunk-2", "doc-2", "from document two", _vector(1.0)),
        ]
    )

    results = repository.similarity_search(
        _vector(1.0), top_k=10, filters={"document_id": "doc-2"}
    )

    assert [result.chunk.id for result in results] == ["chunk-2"]


def test_malicious_text_is_stored_as_data(repository: MariaDBRepository) -> None:
    """Injection payloads in document text are persisted, never executed.

    Args:
        repository: Repository bound to the test schema.
    """
    payload = "' OR '1'='1'; DROP TABLE chunks; --"
    repository.save_document(
        Document(id="doc-1", source_filename=payload, content_hash="h")
    )
    repository.save_chunks_with_embeddings(
        [_embedded("chunk-1", "doc-1", payload, _vector(1.0))]
    )

    results = repository.similarity_search(_vector(1.0), top_k=1)

    assert results[0].chunk.text == payload


def test_injection_in_a_filter_value_matches_nothing(
    repository: MariaDBRepository,
) -> None:
    """A filter value is bound as a parameter, so it cannot alter the query.

    Args:
        repository: Repository bound to the test schema.
    """
    repository.save_document(
        Document(id="doc-1", source_filename="a.pdf", content_hash="h")
    )
    repository.save_chunks_with_embeddings(
        [_embedded("chunk-1", "doc-1", "body", _vector(1.0))]
    )

    results = repository.similarity_search(
        _vector(1.0), top_k=10, filters={"document_id": "' OR '1'='1"}
    )

    assert results == []


def test_unknown_filter_key_is_rejected(repository: MariaDBRepository) -> None:
    """Only allow-listed fields may be filtered on.

    Args:
        repository: Repository bound to the test schema.
    """
    with pytest.raises(ValueError, match="Unsupported search filter"):
        repository.similarity_search(_vector(1.0), top_k=1, filters={"text": "x"})


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_out_of_range_top_k_is_rejected(
    repository: MariaDBRepository, top_k: int
) -> None:
    """Result counts are bounded, so one query cannot exhaust resources.

    Args:
        repository: Repository bound to the test schema.
        top_k: An out-of-range result count.
    """
    with pytest.raises(ValueError, match="top_k must be between"):
        repository.similarity_search(_vector(1.0), top_k=top_k)


def test_wrong_dimension_query_is_rejected(repository: MariaDBRepository) -> None:
    """A query vector of the wrong width fails before reaching the database.

    Args:
        repository: Repository bound to the test schema.
    """
    with pytest.raises(ValueError, match="384-dimensional"):
        repository.similarity_search((1.0, 0.0), top_k=1)


def test_wrong_dimension_embedding_is_rejected(repository: MariaDBRepository) -> None:
    """A mismatched embedding is refused rather than silently truncated.

    Args:
        repository: Repository bound to the test schema.
    """
    repository.save_document(
        Document(id="doc-1", source_filename="a.pdf", content_hash="h")
    )

    with pytest.raises(ValueError, match="384-dimensional"):
        repository.save_chunks_with_embeddings(
            [_embedded("chunk-1", "doc-1", "body", (1.0, 0.0))]
        )


def test_duplicate_documents_are_detectable(repository: MariaDBRepository) -> None:
    """A stored document can be recognised again by its content hash.

    Args:
        repository: Repository bound to the test schema.
    """
    assert not repository.document_exists("deadbeef")

    repository.save_document(
        Document(id="doc-1", source_filename="paper.pdf", content_hash="deadbeef")
    )

    assert repository.document_exists("deadbeef")
    assert not repository.document_exists("other-hash")


def test_empty_writes_are_no_ops(repository: MariaDBRepository) -> None:
    """Saving nothing is harmless and touches no transaction.

    Args:
        repository: Repository bound to the test schema.
    """
    repository.save_pages([])
    repository.save_chunks_with_embeddings([])
