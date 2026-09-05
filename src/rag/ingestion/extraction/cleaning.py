"""Cleaning for text extracted from PDFs.

Encoding repair is delegated to :mod:`ftfy`, which fixes mojibake, control
characters, and Unicode inconsistencies far more thoroughly than hand-written
normalisation would. The layout-aware extractor already resolves the other
classic PDF pathologies — de-hyphenating words split across lines, rejoining
wrapped paragraphs, and folding ligatures — so none of that is repeated here.

What remains is page furniture: the running heads, footers, page numbers, and
publisher stamps that surround the actual content of a paper. Left in place,
they are chunked and embedded alongside real text, and because they recur on
every page they crowd retrieval results with near-identical noise.

Three mechanisms remove them, because no single rule covers the cases seen in
practice:

1. **Structural furniture** — an edge line with almost no letters once digits
   and punctuation are removed. Catches bare page numbers (``2``), combined
   stamps (``**212** Page 2 of 11``), and publisher artefacts
   (``Vol.:(0123456789)``), none of which repeat identically.
2. **Repeated furniture** — an edge line whose digit-masked form recurs on most
   pages. Catches running heads such as
   ``Behavior Research Methods (2025) 57:212``, and masking digits means a head
   carrying a page number still matches itself across pages.
3. **Known artefacts** — a short, explicit list of markers the extractor or
   publisher inserts, such as the HTML comments wrapping figure text.

All three act only on lines near a page's edges, never in the body. Repeated
sentences *within* the body are deliberately left alone: scientific writing
restates definitions and findings legitimately, and removing repeated body
lines would destroy content to solve a problem that only exists at the margins.
Headings and table rows are exempt everywhere, so section structure and tables
survive intact for the chunker.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import ftfy

#: A figure's embedded text, delimited by the extractor's own markers. The
#: whole block is discarded: it holds axis labels, legend entries, and tick
#: values ("0.6<br>Model<br>PPO-ptx<br>0.4<br>SFT"), which read as fragments
#: rather than prose and embed into vectors that match numeric queries without
#: carrying any answer.
_PICTURE_TEXT = re.compile(
    r"<!--\s*Start of picture text\s*-->.*?<!--\s*End of picture text\s*-->",
    re.DOTALL | re.IGNORECASE,
)

#: Any remaining HTML comment, kept as markup noise rather than content.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Three or more consecutive newlines, i.e. more than one blank line.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

#: A Markdown table row. Exempt from every furniture rule.
_TABLE_ROW = re.compile(r"^\s*\|")

#: A Markdown heading. Exempt from every furniture rule, since a numbered
#: section title such as "# 4 Direct Preference Optimization" would otherwise
#: look like structural furniture.
_HEADING = re.compile(r"^\s*#{1,6}\s")

#: Markdown emphasis and inline-code markers, ignored when comparing lines.
_EMPHASIS = re.compile(r"[*_`~]+")

#: Everything that is not a letter, used to measure a line's real content.
_NON_LETTERS = re.compile(r"[^a-z]+")

#: Runs of digits, masked so that a running head carrying a page number
#: compares equal to the same head on another page.
_DIGITS = re.compile(r"\d+")

#: Publisher and extractor artefacts that appear too few times for frequency
#: analysis to catch. Deliberately short; extend it as new sources demand.
_KNOWN_ARTEFACTS = (
    re.compile(r"^vol\.?:?\(\d+\)$", re.IGNORECASE),
    re.compile(r"^\s*(this|the)\s+(preprint|manuscript).{0,60}$", re.IGNORECASE),
    re.compile(r"^downloaded\s+from\b.{0,80}$", re.IGNORECASE),
    re.compile(r"^arxiv:\d{4}\.\d{4,5}v\d+\s+\[[a-z\-.]+\]", re.IGNORECASE),
)

#: Number of lines at each end of a page treated as the page's edges.
_EDGE_LINES = 3

#: Most letters a line may retain, after digits and punctuation are stripped,
#: and still count as structural furniture rather than prose. Real furniture
#: retains very little ("Page 4 of 8" leaves "pageof"), so the bar is set low
#: enough that a short sentence is never mistaken for it.
_MAX_FURNITURE_LETTERS = 8

#: Longest a line can be and still be considered furniture at all.
_MAX_FURNITURE_LENGTH = 120

#: Fraction of pages a line must recur on to count as a running head.
_REPEAT_PAGE_RATIO = 0.6

#: Fewest pages needed before repetition means anything.
_MIN_PAGES_FOR_REPEATS = 4


def repair_text(text: str) -> str:
    """Repair encoding damage and normalise Unicode.

    Args:
        text: The text to repair.

    Returns:
        The repaired text, NFKC-normalised so that presentation variants such
        as ligatures and full-width forms embed identically to their ordinary
        equivalents.
    """
    return ftfy.fix_text(text, normalization="NFKC")


def drop_picture_text(text: str) -> str:
    """Remove the text the extractor recovered from inside figures.

    Chart axis labels and legend entries survive extraction as runs of
    disconnected fragments. They are not prose, they answer no question, and
    they embed into vectors that still match queries mentioning the model names
    or numbers they contain, so the whole block is discarded rather than kept
    as low-quality content.

    Args:
        text: The text to clean.

    Returns:
        The text without figure-embedded text blocks.
    """
    return _PICTURE_TEXT.sub("", text)


def strip_html_comments(text: str) -> str:
    """Remove any remaining HTML comments.

    Args:
        text: The text to clean.

    Returns:
        The text without HTML comments.
    """
    return _HTML_COMMENT.sub("", text)


def tidy_whitespace(text: str) -> str:
    """Trim line-trailing whitespace and collapse runs of blank lines.

    Args:
        text: The text to tidy.

    Returns:
        The text with at most one blank line between blocks.
    """
    trimmed = "\n".join(line.rstrip() for line in text.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", trimmed).strip()


def clean_page_text(text: str) -> str:
    """Apply the per-page cleaning steps in order.

    Args:
        text: The extracted text of a single page.

    Returns:
        The cleaned page text.
    """
    return tidy_whitespace(strip_html_comments(drop_picture_text(repair_text(text))))


def _is_exempt(line: str) -> bool:
    """Report whether a line is protected from every furniture rule.

    Args:
        line: The line to test.

    Returns:
        ``True`` for headings and table rows, which are always content.
    """
    return bool(_HEADING.match(line) or _TABLE_ROW.match(line))


def _canonical(line: str) -> str:
    """Reduce a line to a form comparable across pages.

    Emphasis markers are dropped, digits are masked, and whitespace is
    collapsed, so that a running head differing only by its page number
    compares equal to itself on every page.

    Args:
        line: The line to canonicalise.

    Returns:
        The canonical form.
    """
    without_emphasis = _EMPHASIS.sub("", line).strip().lower()
    return " ".join(_DIGITS.sub("#", without_emphasis).split())


def _letter_content(line: str) -> str:
    """Extract just the letters of a line.

    Args:
        line: The line to reduce.

    Returns:
        The line's lowercase letters, with everything else removed.
    """
    return _NON_LETTERS.sub("", _EMPHASIS.sub("", line).lower())


def _is_structural_furniture(line: str) -> bool:
    """Report whether a line is page furniture by its shape alone.

    A line consisting of a page number, a folio, or a publisher stamp retains
    almost no letters once digits and punctuation are removed, whereas even a
    short sentence retains many.

    Args:
        line: The line to test.

    Returns:
        ``True`` if the line looks like page furniture.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_FURNITURE_LENGTH:
        return False
    if any(pattern.match(stripped) for pattern in _KNOWN_ARTEFACTS):
        return True
    return len(_letter_content(stripped)) <= _MAX_FURNITURE_LETTERS


def _edge_positions(lines: Sequence[str]) -> set[int]:
    """Identify which lines sit at a page's edges.

    Blank lines are skipped when locating the edges but keep their positions,
    so that removing furniture never disturbs the blank lines separating
    paragraphs — the chunker relies on those to find block boundaries.

    Args:
        lines: The page's lines, blanks included.

    Returns:
        Indices into ``lines`` of the non-blank lines at either end.
    """
    filled = [index for index, line in enumerate(lines) if line.strip()]
    if not filled:
        return set()
    # The window never grows to cover a whole page: a sparse page must keep a
    # protected middle, or its only content would be eligible for removal.
    edge = min(_EDGE_LINES, max(1, len(filled) // 3))
    return set(filled[:edge]) | set(filled[-edge:])


def find_running_boilerplate(pages: Sequence[str]) -> frozenset[str]:
    """Identify running heads and footers shared across pages.

    Args:
        pages: The cleaned text of every page, in source order.

    Returns:
        Canonical forms, from :func:`_canonical`, of lines that recur at the
        edges of most pages. Empty for documents too short for repetition to
        be meaningful.
    """
    if len(pages) < _MIN_PAGES_FOR_REPEATS:
        return frozenset()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.split("\n")
        edges = {
            _canonical(lines[index])
            for index in _edge_positions(lines)
            if not _is_exempt(lines[index])
            and len(lines[index].strip()) <= _MAX_FURNITURE_LENGTH
        }
        counts.update(edges - {""})

    threshold = len(pages) * _REPEAT_PAGE_RATIO
    return frozenset(line for line, count in counts.items() if count >= threshold)


def strip_page_furniture(text: str, boilerplate: frozenset[str]) -> str:
    """Remove page furniture from a page's edges.

    Args:
        text: The cleaned page text.
        boilerplate: Canonical running-head forms from
            :func:`find_running_boilerplate`.

    Returns:
        The page text with its furniture removed.
    """
    lines = text.split("\n")
    edges = _edge_positions(lines)

    kept = [
        line
        for index, line in enumerate(lines)
        if index not in edges
        or _is_exempt(line)
        or not (_is_structural_furniture(line) or _canonical(line) in boilerplate)
    ]
    cleaned = tidy_whitespace("\n".join(kept))
    # Furniture removal must never empty a page. A page short enough for every
    # line to look like furniture is far more likely to be a sparse title or
    # figure page than a page of pure boilerplate.
    return cleaned or tidy_whitespace(text)
