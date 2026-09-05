"""Tests for the chunkers and the contamination filters around them."""

from __future__ import annotations

import pytest

from rag.domain.models import Page
from rag.ingestion.chunking.pipeline import (
    REFERENCES_SECTION,
    SECTION_TYPE_KEY,
    ChunkingPipeline,
)
from rag.ingestion.chunking.splitters import (
    MarkdownHeaderChunker,
    RecursiveChunker,
    normalise_heading,
)


def _page(text: str, *, number: int = 1, ocr: bool = False) -> Page:
    """Build a page for a test.

    Args:
        text: The page's text.
        number: The page number.
        ocr: Whether the page's text came from OCR.

    Returns:
        The page.
    """
    return Page(document_id="doc-1", page_number=number, text=text, ocr_extracted=ocr)


def test_headings_lose_their_markdown_formatting() -> None:
    """Section names read as plain wording, not as Markdown source."""
    assert normalise_heading("**References**") == "References"


def test_section_trail_is_recorded() -> None:
    """A chunk knows the heading path that leads to it."""
    page = _page("# Methods\nIntro text.\n## Participants\nForty adults took part.")

    chunks = MarkdownHeaderChunker().chunk([page])

    assert chunks[-1].section == "Methods > Participants"
    assert chunks[-1].headers == ("Methods", "Participants")


def test_text_before_the_first_heading_is_kept() -> None:
    """Abstracts and front matter are not discarded."""
    page = _page("An abstract with no heading above it.\n\n# Introduction\nBody.")

    chunks = MarkdownHeaderChunker().chunk([page])

    assert "An abstract with no heading above it." in chunks[0].text


def test_section_continues_across_a_page_break() -> None:
    """A section spanning pages keeps its heading on the later pages."""
    pages = [
        _page("# Methods\nFirst half of the section.", number=1),
        _page("Second half, with no heading of its own.", number=2),
    ]

    chunks = MarkdownHeaderChunker().chunk(pages)

    continuation = next(c for c in chunks if c.page_number == 2)
    assert continuation.headers == ("Methods",)


def test_ancestors_survive_a_page_break() -> None:
    """A subsection starting a new page keeps the sections containing it."""
    pages = [
        _page("# Methods\n## Design\nText.", number=1),
        _page("### Participants\nForty adults took part.", number=2),
    ]

    chunks = MarkdownHeaderChunker().chunk(pages)

    continuation = next(c for c in chunks if c.page_number == 2)
    assert continuation.headers == ("Methods", "Design", "Participants")


def test_a_new_top_level_heading_closes_previous_subsections() -> None:
    """Declaring a heading clears everything nested under the previous one."""
    pages = [
        _page("# Methods\n## Design\nText.", number=1),
        _page("# Results\nFindings follow.", number=2),
    ]

    chunks = MarkdownHeaderChunker().chunk(pages)

    assert next(c for c in chunks if c.page_number == 2).headers == ("Results",)


def test_page_provenance_is_preserved() -> None:
    """Each chunk records the page it came from."""
    pages = [_page("# A\nOne.", number=3), _page("# B\nTwo.", number=9)]

    assert {c.page_number for c in MarkdownHeaderChunker().chunk(pages)} == {3, 9}


def test_ocr_provenance_reaches_the_chunk() -> None:
    """A chunk from an OCR'd page is marked, so it can be filtered later."""
    chunks = MarkdownHeaderChunker().chunk([_page("# A\nText.", ocr=True)])

    assert chunks[0].metadata["ocr_extracted"] is True


def test_oversized_chunks_are_split_with_provenance_intact() -> None:
    """Splitting by size does not lose the section a passage came from."""
    page = _page("# Methods\n" + "Sentence about the study. " * 200)
    section = MarkdownHeaderChunker().chunk([page])[0]

    pieces = RecursiveChunker(chunk_size=500, chunk_overlap=50).split_oversized(section)

    assert len(pieces) > 1
    assert all(piece.section == "Methods" for piece in pieces)
    assert all(piece.page_number == 1 for piece in pieces)


def test_recursive_chunker_works_standalone() -> None:
    """Size-only chunking is usable on its own, without structure."""
    page = _page("Sentence about the study. " * 100)

    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=30).chunk([page])

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_chunks_within_budget_are_left_alone() -> None:
    """A chunk that already fits is returned unchanged."""
    section = MarkdownHeaderChunker().chunk([_page("# A\nShort body text.")])[0]

    assert RecursiveChunker(chunk_size=5000).split_oversized(section) == [section]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (100, 100), (100, 200), (100, -1)],
)
def test_invalid_chunk_sizes_are_rejected(chunk_size: int, chunk_overlap: int) -> None:
    """Nonsensical size settings fail at construction.

    Args:
        chunk_size: The target chunk size under test.
        chunk_overlap: The overlap under test.
    """
    with pytest.raises(ValueError, match="chunk_"):
        RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_reference_sections_are_dropped_by_default() -> None:
    """A bibliography does not reach the index."""
    page = _page(
        "# Discussion\nThe findings suggest a robust effect across conditions.\n"
        "# References\nSmith, J. (2020). A paper about things. Journal, 1, 1-10."
    )

    chunks = ChunkingPipeline().chunk([page])

    assert all("References" not in (c.section or "") for c in chunks)
    assert any("findings suggest" in c.text for c in chunks)


def test_reference_sections_can_be_retained_and_are_tagged() -> None:
    """Bibliographies are kept when asked for, and marked as such."""
    page = _page(
        "# Discussion\nThe findings suggest a robust effect across conditions.\n"
        "# References\nSmith, J. (2020). A paper about things. Journal, 1, 1-10."
    )

    chunks = ChunkingPipeline(drop_references=False).chunk([page])

    references = [c for c in chunks if c.section == "References"]
    assert references
    assert references[0].metadata[SECTION_TYPE_KEY] == REFERENCES_SECTION


@pytest.mark.parametrize(
    "heading", ["References", "Bibliography", "7 References", "WORKS CITED"]
)
def test_reference_headings_are_recognised(heading: str) -> None:
    """The common spellings of a bibliography heading are all caught.

    Args:
        heading: The heading spelling under test.
    """
    page = _page(f"# {heading}\nSmith, J. (2020). A paper. Journal of Things, 1, 1-10.")

    assert ChunkingPipeline().chunk([page]) == []


def test_a_section_named_like_a_reference_to_something_is_kept() -> None:
    """Only a bibliography heading matches, not prose that mentions one."""
    page = _page(
        "# References to prior work\n"
        "This section discusses how earlier studies approached the problem."
    )

    assert ChunkingPipeline().chunk([page])


def test_fragments_are_dropped() -> None:
    """A chunk that is just a stray heading does not reach the index."""
    page = _page("# A\n\n# B\n\n# C")

    assert ChunkingPipeline(min_chunk_chars=40).chunk([page]) == []


def test_negative_minimum_length_is_rejected() -> None:
    """A nonsensical filter setting fails at construction."""
    with pytest.raises(ValueError, match="min_chunk_chars"):
        ChunkingPipeline(min_chunk_chars=-1)


def test_tables_stay_whole_when_they_fit() -> None:
    """A table within budget is never broken across chunks."""
    table = "\n".join(
        ["| Condition | N |", "|---|---|"]
        + [f"| Group {i} | {i * 10} |" for i in range(6)]
    )
    page = _page(f"# Results\n\n{table}\n\nProse following the table.")

    chunks = ChunkingPipeline(chunk_size=1200).chunk([page])

    assert any(c.text.count("| Group") == 6 for c in chunks)
