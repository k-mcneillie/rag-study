"""Tests for PDF text cleaning and page-furniture removal."""

from __future__ import annotations

from rag.ingestion.extraction.cleaning import (
    clean_page_text,
    drop_picture_text,
    find_running_boilerplate,
    repair_text,
    strip_page_furniture,
    tidy_whitespace,
)


def test_mojibake_is_repaired() -> None:
    """Text mis-decoded during extraction is restored."""
    assert repair_text("naÃ¯ve rÃ©sumÃ©") == "naïve résumé"


def test_ligatures_are_folded() -> None:
    """Presentation forms fold so they embed as ordinary words."""
    assert repair_text("efﬁcient ﬂow") == "efficient flow"


def test_picture_text_blocks_are_discarded() -> None:
    """Chart axis labels and legends are removed rather than embedded."""
    text = (
        "Real prose before.\n"
        "<!-- Start of picture text -->\n"
        "0.6<br>Model<br>PPO-ptx<br>0.4<br>SFT<br>1.3B 6B 175B<br>\n"
        "<!-- End of picture text -->\n"
        "Real prose after."
    )

    cleaned = drop_picture_text(text)

    assert "PPO-ptx" not in cleaned
    assert "Real prose before." in cleaned
    assert "Real prose after." in cleaned


def test_blank_lines_between_blocks_are_preserved() -> None:
    """Paragraph separation survives, since the splitter relies on it."""
    assert tidy_whitespace("A\n\n\n\n\nB") == "A\n\nB"


def test_clean_page_text_leaves_tables_intact() -> None:
    """Table rows pass through cleaning unchanged."""
    table = "| Condition | N |\n|---|---|\n| Control | 40 |"

    assert clean_page_text(table) == table


def _paginate(body: str, count: int) -> list[str]:
    """Build a document whose pages share a running head and page numbers.

    Pages carry a realistic amount of body text, because furniture is only
    removed from a page's edges and a page must be long enough to have a
    middle that is protected from removal.

    Args:
        body: Body text placed on every page.
        count: How many pages to build.

    Returns:
        The generated pages.
    """
    return [
        "\n".join(
            [
                "Behavior Research Methods (2025) 57:212",
                f"Page {n} of {count}",
                *(f"Opening line {i} of page {n}." for i in range(4)),
                body,
                *(f"Closing line {i} of page {n}." for i in range(4)),
                str(n),
            ]
        )
        for n in range(1, count + 1)
    ]


def test_running_heads_are_detected_despite_page_numbers() -> None:
    """Digit masking lets a head carrying a page number match across pages."""
    pages = _paginate("Substantive content that differs little.", 8)

    boilerplate = find_running_boilerplate(pages)

    assert any("behavior research methods" in line for line in boilerplate)


def test_page_furniture_is_removed_from_pages() -> None:
    """Running heads, folios, and bare page numbers all disappear."""
    pages = _paginate("Substantive content worth keeping in the index.", 8)
    boilerplate = find_running_boilerplate(pages)

    cleaned = strip_page_furniture(pages[3], boilerplate)

    assert "Substantive content worth keeping" in cleaned
    assert "Behavior Research Methods" not in cleaned
    assert "Page 4 of 8" not in cleaned
    assert not cleaned.endswith("4")


def _page_with_body(first: str = "", last: str = "") -> str:
    """Build a single realistic page, optionally topped and tailed.

    Args:
        first: Line to place above the body.
        last: Line to place below the body.

    Returns:
        The generated page text.
    """
    body = "\n".join(f"Body line {i} carrying real content." for i in range(8))
    return "\n".join(part for part in (first, body, last) if part)


def test_bare_page_number_is_removed_without_repetition() -> None:
    """A page number is furniture by shape, even when it never repeats."""
    cleaned = strip_page_furniture(_page_with_body(last="7"), frozenset())

    assert not cleaned.endswith("7")
    assert "Body line 7 carrying real content." in cleaned


def test_publisher_stamp_is_removed() -> None:
    """A one-off publisher artefact is caught by shape, not frequency."""
    cleaned = strip_page_furniture(
        _page_with_body(first="Vol.:(0123456789)"), frozenset()
    )

    assert "Vol." not in cleaned


def test_furniture_removal_never_empties_a_page() -> None:
    """A sparse page keeps its content rather than being wiped out."""
    assert strip_page_furniture("Short title", frozenset()) == "Short title"


def test_headings_are_never_treated_as_furniture() -> None:
    """A numbered section heading survives, despite looking numeric."""
    page = "# 4 Direct Preference Optimization\nBody text follows the heading."

    assert "# 4 Direct Preference Optimization" in strip_page_furniture(
        page, frozenset()
    )


def test_table_rows_are_never_treated_as_furniture() -> None:
    """A numeric table row at a page edge is content, not furniture."""
    page = "| 12 | 44 |\nSurrounding prose that anchors the table.\n| 13 | 45 |"

    cleaned = strip_page_furniture(page, frozenset())

    assert "| 12 | 44 |" in cleaned
    assert "| 13 | 45 |" in cleaned


def test_short_documents_are_left_alone() -> None:
    """Repetition means nothing across a handful of pages."""
    assert find_running_boilerplate(["Shared line\nA", "Shared line\nB"]) == frozenset()


def test_repeated_body_lines_are_kept() -> None:
    """Prose repeated in a page's body survives, unlike edge furniture.

    Scientific writing legitimately restates definitions and findings, so
    repetition is only treated as boilerplate at a page's edges.
    """
    repeated = "The model was trained to convergence."
    pages = [
        "\n".join(
            [
                "Running Head",
                *(f"Opening sentence {i} of page {n}." for i in range(3)),
                repeated,
                *(f"Closing sentence {i} of page {n}." for i in range(3)),
                str(n),
            ]
        )
        for n in range(1, 9)
    ]
    boilerplate = find_running_boilerplate(pages)

    cleaned = strip_page_furniture(pages[2], boilerplate)

    assert repeated in cleaned
    assert "Running Head" not in cleaned
