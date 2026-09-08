"""``POST /documents`` ingests one uploaded PDF synchronously."""

from __future__ import annotations

from pathlib import Path

from rag.ingestion import IngestionResult
from tests.service.conftest import API_KEY, Harness

_AUTH = {"X-API-Key": API_KEY}


def _upload(harness: Harness, content: bytes, name: str = "paper.pdf") -> object:
    """POST one file to the documents endpoint.

    Args:
        harness: The test app.
        content: The file bytes.
        name: The upload filename.

    Returns:
        The HTTP response.
    """
    return harness.client.post(
        "/documents",
        files={"file": (name, content, "application/pdf")},
        headers=_AUTH,
    )


def test_a_new_document_is_ingested(harness: Harness) -> None:
    """A stored document returns 201 with its id and chunk count.

    Args:
        harness: The test app.
    """
    response = _upload(harness, b"%PDF-1.4 ...")

    assert response.status_code == 201
    assert response.json() == {
        "status": "ingested",
        "document_id": "doc-9",
        "chunk_count": 3,
    }
    assert len(harness.ingestion.calls) == 1
    assert harness.ingestion.calls[0].suffix == ".pdf"


def test_a_duplicate_returns_already_indexed(harness: Harness) -> None:
    """Identical content already stored returns 200, not 201.

    Args:
        harness: The test app.
    """
    harness.ingestion.result = IngestionResult(source=Path("x.pdf"), skipped=True)

    response = _upload(harness, b"%PDF-1.4 ...")

    assert response.status_code == 200
    assert response.json() == {
        "status": "already_indexed",
        "document_id": None,
        "chunk_count": None,
    }


def test_an_unreadable_document_is_a_422(harness: Harness) -> None:
    """A document that yields no text is reported, not raised.

    Args:
        harness: The test app.
    """
    harness.ingestion.result = IngestionResult(
        source=Path("x.pdf"), error="x.pdf yielded no usable text."
    )

    response = _upload(harness, b"%PDF-1.4 ...")

    assert response.status_code == 422
    assert response.json()["detail"] == "x.pdf yielded no usable text."


def test_a_non_pdf_is_refused(harness: Harness) -> None:
    """A non-PDF upload is refused before any ingestion runs.

    Args:
        harness: The test app.
    """
    response = _upload(harness, b"hello", name="notes.txt")

    assert response.status_code == 415
    assert harness.ingestion.calls == []


def test_an_oversize_upload_is_refused(harness: Harness) -> None:
    """A file past the byte limit is refused with 413.

    Args:
        harness: The test app.
    """
    harness.settings.max_document_bytes = 8  # type: ignore[attr-defined]

    response = _upload(harness, b"%PDF-1.4 and then some more bytes")

    assert response.status_code == 413
    assert harness.ingestion.calls == []
