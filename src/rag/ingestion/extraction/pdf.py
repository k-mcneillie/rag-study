"""PDF extraction.

Layout analysis is delegated to ``pymupdf4llm``, which reads a page's geometry
to recover reading order, so two-column papers, templated journal articles, and
mixed layouts all come back in the order a human would read them. Tables are
returned as Markdown tables and mathematics as readable inline text. Its layout
and table models ship inside the installed package, so nothing is fetched at
runtime.

Mathematics is preserved as text rather than reconstructed as LaTeX. PDFs
record the glyphs of an equation, not its structure, and guessing that
structure back would be both unreliable and far more machinery than retrieval
needs — the extracted form still carries the symbols and identifiers a query
can match.

Every PDF is treated as untrusted input: the file is validated before it is
opened, size and page-count limits are enforced, and a failure to parse raises
:class:`~rag.ingestion.interfaces.ExtractionError` for one document rather than
ending a whole ingestion run.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from rag.domain.models import Document, ExtractedDocument, Page
from rag.ingestion.extraction.cleaning import (
    clean_page_text,
    find_running_boilerplate,
    strip_page_furniture,
)
from rag.ingestion.interfaces import BaseExtractor, ExtractionError

logger = logging.getLogger(__name__)

#: Every PDF begins with this signature.
_PDF_MAGIC = b"%PDF-"

#: Below this many characters, a page's own text layer is treated as absent,
#: which means any text recovered from it came from OCR.
_TEXT_LAYER_THRESHOLD = 32

#: Document metadata worth keeping. Everything else in a PDF's metadata
#: dictionary is ignored rather than carried around untrusted.
_METADATA_KEYS = ("title", "author", "subject", "creationDate")


class PDFExtractor(BaseExtractor):
    """Extracts pages from a PDF as layout-aware Markdown.

    Attributes:
        max_bytes: Largest file this extractor will open.
        max_pages: Most pages it will extract from one document.
        use_ocr: Whether to OCR pages that have no text layer.
    """

    def __init__(
        self,
        *,
        max_bytes: int = 100 * 1024 * 1024,
        max_pages: int = 2000,
        use_ocr: bool = True,
    ) -> None:
        """Initialise the extractor.

        Args:
            max_bytes: Largest file to accept, bounding memory and parsing
                cost for a single document.
            max_pages: Most pages to extract from one document.
            use_ocr: Whether to OCR pages with no text layer. Requires the
                Tesseract binary to be installed; when it is unavailable,
                scanned pages yield no text rather than failing.
        """
        self.max_bytes = max_bytes
        self.max_pages = max_pages
        self.use_ocr = use_ocr

    def extract(self, source: Path) -> ExtractedDocument:
        """Extract a PDF's pages as cleaned Markdown.

        Args:
            source: Path to the PDF.

        Returns:
            The document record together with its extracted pages.

        Raises:
            ExtractionError: If the file is missing, is not a PDF, exceeds a
                configured limit, or cannot be parsed.
        """
        self._validate(source)
        content_hash = _hash_file(source)

        try:
            with pymupdf.open(source) as document:
                if document.page_count > self.max_pages:
                    raise ExtractionError(
                        f"{source.name} has {document.page_count} pages, "
                        f"exceeding the limit of {self.max_pages}."
                    )
                raw_pages = self._extract_pages(document, source)
                metadata = _document_metadata(document)
        except ExtractionError:
            raise
        except Exception as exc:
            # PyMuPDF surfaces malformed files as a range of exception types;
            # the detail is logged, while the caller gets a safe summary.
            logger.warning("Failed to parse %s: %s", source.name, type(exc).__name__)
            raise ExtractionError(
                f"{source.name} could not be parsed as a PDF."
            ) from exc

        record = Document(
            source_filename=source.name,
            content_hash=content_hash,
            metadata=metadata,
        )
        return ExtractedDocument(
            document=record, pages=_build_pages(record.id, raw_pages)
        )

    def _validate(self, source: Path) -> None:
        """Check a source file before opening it.

        Args:
            source: Path to the candidate PDF.

        Raises:
            ExtractionError: If the path is not a readable PDF within limits.
        """
        if not source.is_file():
            raise ExtractionError(f"{source.name} is not a readable file.")

        size = source.stat().st_size
        if size > self.max_bytes:
            raise ExtractionError(
                f"{source.name} is {size} bytes, "
                f"exceeding the limit of {self.max_bytes}."
            )
        if size == 0:
            raise ExtractionError(f"{source.name} is empty.")

        # Trust the file's own signature rather than its extension, which is
        # attacker-controlled and says nothing about the bytes on disk.
        with source.open("rb") as handle:
            if handle.read(len(_PDF_MAGIC)) != _PDF_MAGIC:
                raise ExtractionError(f"{source.name} is not a PDF.")

    def _extract_pages(
        self, document: pymupdf.Document, source: Path
    ) -> list[tuple[str, bool]]:
        """Extract Markdown and OCR status for each page.

        Args:
            document: The opened PDF.
            source: Path to the PDF, used for log messages.

        Returns:
            One ``(markdown, ocr_extracted)`` pair per page, in source order.
        """
        # Whether each page carries its own text layer is decided before
        # extraction, so a page whose text could only have come from OCR can
        # be flagged for downstream callers.
        has_text_layer = [
            len(document[number].get_text().strip()) >= _TEXT_LAYER_THRESHOLD
            for number in range(document.page_count)
        ]

        chunks = pymupdf4llm.to_markdown(
            document, page_chunks=True, use_ocr=self.use_ocr, show_progress=False
        )
        if len(chunks) != len(has_text_layer):
            logger.warning(
                "Page count mismatch for %s: %d extracted, %d expected.",
                source.name,
                len(chunks),
                len(has_text_layer),
            )

        return [
            (str(chunk.get("text", "")), not native)
            for chunk, native in zip(chunks, has_text_layer, strict=False)
        ]


def _build_pages(
    document_id: str, raw_pages: list[tuple[str, bool]]
) -> tuple[Page, ...]:
    """Clean extracted page text and assemble the page records.

    Boilerplate detection needs every page at once, so cleaning happens here
    rather than page by page during extraction.

    Args:
        document_id: Identifier of the owning document.
        raw_pages: One ``(markdown, ocr_extracted)`` pair per page.

    Returns:
        The cleaned pages, omitting any that cleaned down to nothing.
    """
    cleaned = [clean_page_text(text) for text, _ in raw_pages]
    boilerplate = find_running_boilerplate(cleaned)

    pages = []
    for number, (text, (_, ocr_extracted)) in enumerate(
        zip(cleaned, raw_pages, strict=True), start=1
    ):
        body = strip_page_furniture(text, boilerplate)
        if not body:
            continue
        pages.append(
            Page(
                document_id=document_id,
                page_number=number,
                text=body,
                ocr_extracted=ocr_extracted,
            )
        )
    return tuple(pages)


def _document_metadata(document: pymupdf.Document) -> dict[str, Any]:
    """Collect the document metadata worth keeping.

    The values are untrusted document content and are stored as data only.

    Args:
        document: The opened PDF.

    Returns:
        The selected metadata entries that carry a value.
    """
    metadata = document.metadata or {}
    return {key: str(metadata[key]) for key in _METADATA_KEYS if metadata.get(key)}


def _hash_file(source: Path, *, block_size: int = 1 << 20) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        source: Path to the file.
        block_size: Number of bytes to read at a time, so that large documents
            are never held in memory in full.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()
