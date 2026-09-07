"""Tests for semantic chunking.

A fake embedder supplies fixed vectors, so the boundary logic is tested
deterministically and without a model. That the chunker can be tested this way
is the point of it depending on a two-method capability rather than on an
embedder.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag.domain.models import Chunk, Page
from rag.ingestion.chunking.semantic import SemanticChunker, TextEmbedder

#: Two orthogonal directions, so "same topic" and "different topic" are exact.
TOPIC_A = (1.0, 0.0)
TOPIC_B = (0.0, 1.0)


class TopicEmbedder:
    """Embeds by keyword: sentences mentioning 'beta' are a different topic."""

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Assign each text one of two orthogonal vectors.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per text.
        """
        return [TOPIC_B if "beta" in text.lower() else TOPIC_A for text in texts]


def _page(text: str, *, ocr: bool = False) -> Page:
    """Build a page for a test.

    Args:
        text: The page's text.
        ocr: Whether its text came from OCR.

    Returns:
        The page.
    """
    return Page(document_id="doc-1", page_number=2, text=text, ocr_extracted=ocr)


@pytest.fixture
def chunker() -> SemanticChunker:
    """Provide a chunker over the two-topic fake embedder.

    Returns:
        The chunker under test.
    """
    return SemanticChunker(TopicEmbedder(), min_chunk_chars=10)


def test_the_capability_is_satisfiable_without_an_embedder() -> None:
    """The chunker's dependency is a shape, not a model."""
    assert isinstance(TopicEmbedder(), TextEmbedder)


def test_a_boundary_is_placed_where_the_subject_changes(
    chunker: SemanticChunker,
) -> None:
    """Sentences about different subjects end up in different chunks.

    Args:
        chunker: The chunker under test.
    """
    page = _page(
        "Alpha concerns the first idea. Alpha continues the first idea here. "
        "Beta introduces a separate matter. Beta develops that separate matter."
    )

    chunks = chunker.chunk([page])

    assert len(chunks) == 2
    assert "beta" not in chunks[0].text.lower()
    assert "alpha" not in chunks[1].text.lower()


def test_related_sentences_stay_together(chunker: SemanticChunker) -> None:
    """Uniform prose is not split for its own sake.

    Args:
        chunker: The chunker under test.
    """
    page = _page(
        "Alpha concerns the first idea. Alpha continues that idea. "
        "Alpha concludes the same idea."
    )

    assert len(chunker.chunk([page])) == 1


def test_chunks_never_exceed_the_ceiling() -> None:
    """A passage with no topic shift is still bounded.

    The size limit applies regardless of similarity, so uniform prose cannot
    produce one enormous chunk.
    """
    chunker = SemanticChunker(TopicEmbedder(), max_chunk_chars=200, min_chunk_chars=10)
    page = _page(" ".join(["Alpha restates the very same idea again."] * 40))

    assert all(len(chunk.text) <= 200 for chunk in chunker.chunk([page]))


def test_provenance_is_preserved(chunker: SemanticChunker) -> None:
    """Every chunk knows which document and page it came from.

    Args:
        chunker: The chunker under test.
    """
    chunks = chunker.chunk(
        [_page("Alpha one idea. Beta another idea entirely.", ocr=True)]
    )

    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.ocr_extracted for chunk in chunks)


def test_refining_a_section_keeps_its_heading_trail() -> None:
    """Composed with structural chunking, the section survives the split."""
    section = Chunk(
        document_id="doc-1",
        page_number=2,
        text=(
            "Alpha concerns the first idea, at some length so that it exceeds "
            "the configured ceiling for a single chunk of text. "
            "Beta introduces a separate matter, also at length, so that the "
            "two topics cannot fit together within the budget."
        ),
        section="Methods > Design",
        headers=("Methods", "Design"),
    )
    small = SemanticChunker(TopicEmbedder(), max_chunk_chars=120, min_chunk_chars=10)

    pieces = small.refine(section)

    assert len(pieces) > 1
    assert all(piece.section == "Methods > Design" for piece in pieces)
    assert all(piece.headers == ("Methods", "Design") for piece in pieces)


def test_a_section_within_budget_is_left_alone(chunker: SemanticChunker) -> None:
    """Refinement does nothing to a chunk that already fits.

    Args:
        chunker: The chunker under test.
    """
    section = Chunk(document_id="doc-1", page_number=1, text="Short enough.")

    assert chunker.refine(section) == [section]


def test_table_rows_are_kept_whole(chunker: SemanticChunker) -> None:
    """A table row is never split from itself by sentence detection.

    Args:
        chunker: The chunker under test.
    """
    page = _page("| Condition | N |\n|---|---|\n| Control. Treated. | 40 |")

    assert any(
        "| Control. Treated. | 40 |" in chunk.text for chunk in chunker.chunk([page])
    )


def test_a_single_sentence_page_is_handled(chunker: SemanticChunker) -> None:
    """There is nothing to compare, and the page still becomes a chunk.

    Args:
        chunker: The chunker under test.
    """
    assert len(chunker.chunk([_page("Only one sentence here.")])) == 1


def test_an_empty_page_yields_nothing(chunker: SemanticChunker) -> None:
    """A blank page produces no chunks rather than an empty one.

    Args:
        chunker: The chunker under test.
    """
    assert chunker.chunk([_page("   ")]) == []


@pytest.mark.parametrize(
    ("percentile", "max_chars", "min_chars"),
    [
        (0.0, 1200, 100),
        (100.0, 1200, 100),
        (85.0, 0, 100),
        (85.0, 100, 100),
        (85.0, 100, -1),
    ],
)
def test_invalid_settings_are_rejected(
    percentile: float, max_chars: int, min_chars: int
) -> None:
    """Nonsensical configuration fails at construction.

    Args:
        percentile: The breakpoint percentile under test.
        max_chars: The size ceiling under test.
        min_chars: The minimum chunk size under test.
    """
    with pytest.raises(ValueError):
        SemanticChunker(
            TopicEmbedder(),
            breakpoint_percentile=percentile,
            max_chunk_chars=max_chars,
            min_chunk_chars=min_chars,
        )
