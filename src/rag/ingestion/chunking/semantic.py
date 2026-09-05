"""Semantic chunking: cutting where the meaning changes.

Structural chunking cuts at headings, and recursive chunking cuts at a
character budget. Neither notices when a passage changes subject mid-section,
which is exactly where a chunk boundary belongs: a chunk spanning two ideas
embeds as the average of both and matches neither well.

This chunker measures that directly. It splits a page into sentences, embeds
them, and compares each sentence with the one before it. Where consecutive
sentences are unusually dissimilar, the topic has moved and a boundary is
placed. The threshold is a percentile of the distances actually observed in
the document rather than a fixed number, so a densely written paper and a
loosely written one are each judged on their own terms.

**On the dependency rule.** The architecture says a chunker knows nothing
about embeddings, and semantic chunking obviously needs them. The rule is kept
where it matters: this module never imports an embedder, a model, or any
library that loads one. It depends on :class:`TextEmbedder`, a two-method
description of the capability it needs, which the caller supplies. The chunker
still knows nothing about *which* model is used, whether one is local, or how
it is loaded — and it remains testable with a handful of fixed vectors.

The cost is real and worth stating plainly: this embeds every sentence, on top
of embedding every finished chunk. It is not the default for that reason, and
it is most worth using on long, discursive prose where structural headings are
too coarse to be useful.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag.domain.models import Chunk, Page
from rag.ingestion.chunking.splitters import SECTION_SEPARATOR
from rag.ingestion.interfaces import BaseChunker

#: Sentence boundary: terminal punctuation followed by whitespace. Deliberately
#: simple. A full sentence tokeniser would add a dependency and a model to this
#: module, and a boundary placed one sentence out costs very little, because
#: neighbouring sentences are similar by construction.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

#: Lines that must never be split across chunks: table rows and headings carry
#: structure that is meaningless once divided.
_STRUCTURAL_LINE = re.compile(r"^\s*(\||#{1,6}\s)")


@runtime_checkable
class TextEmbedder(Protocol):
    """The embedding capability semantic chunking requires.

    Deliberately narrower than the ingestion embedder: it takes text and
    returns vectors, and says nothing about chunks, models, or batching. Any
    callable object matching this shape will do, which is what keeps this
    module independent of how embeddings are actually produced.
    """

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed a sequence of texts.

        Args:
            texts: The texts to embed.

        Returns:
            One vector per input text, in the same order.
        """
        ...


class SemanticChunker(BaseChunker):
    """Groups adjacent sentences, breaking where the subject changes.

    Attributes:
        breakpoint_percentile: How readily a boundary is placed. Higher values
            split less often.
        max_chunk_chars: Hard ceiling on a chunk, applied regardless of
            similarity.
        min_chunk_chars: Shortest chunk worth emitting on its own.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        breakpoint_percentile: float = 85.0,
        max_chunk_chars: int = 1200,
        min_chunk_chars: int = 100,
    ) -> None:
        """Initialise the chunker.

        Args:
            embedder: Supplies sentence embeddings.
            breakpoint_percentile: Percentile of observed sentence-to-sentence
                distances above which a boundary is placed. Expressed as a
                percentile rather than an absolute distance because what counts
                as a large shift differs between documents.
            max_chunk_chars: Hard ceiling on chunk size, so a passage with no
                clear topic shift still cannot grow without bound.
            min_chunk_chars: Shortest chunk to emit before merging it into the
                next, preventing a stray sentence becoming its own chunk.

        Raises:
            ValueError: If the percentile is outside 0-100, or the sizes are
                not positive and ordered.
        """
        if not 0.0 < breakpoint_percentile < 100.0:
            raise ValueError("breakpoint_percentile must be between 0 and 100.")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive.")
        if not 0 <= min_chunk_chars < max_chunk_chars:
            raise ValueError(
                "min_chunk_chars must be non-negative and below max_chunk_chars."
            )

        self._embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Split pages where their subject matter changes.

        Args:
            pages: The pages to split, in source order.

        Returns:
            The resulting chunks, in document order.
        """
        chunks: list[Chunk] = []
        for page in pages:
            chunks.extend(self._chunk_page(page, headers=()))
        return chunks

    def refine(self, chunk: Chunk) -> list[Chunk]:
        """Split one existing chunk semantically, keeping its provenance.

        This is how semantic chunking composes with structural chunking:
        sections are found first by their headings, and only those too long to
        stand as one passage are divided here, so every piece keeps the section
        trail it came from.

        Args:
            chunk: The chunk to split.

        Returns:
            The chunk unchanged if it is within budget, otherwise its pieces.
        """
        if len(chunk.text) <= self.max_chunk_chars:
            return [chunk]

        page = Page(
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            text=chunk.text,
            ocr_extracted=chunk.ocr_extracted,
        )
        return self._chunk_page(page, headers=chunk.headers, metadata=chunk.metadata)

    def _chunk_page(
        self,
        page: Page,
        *,
        headers: Sequence[str],
        metadata: dict[str, object] | None = None,
    ) -> list[Chunk]:
        """Split one page's text at its semantic boundaries.

        Args:
            page: The page to split.
            headers: Heading trail to record on every resulting chunk.
            metadata: Metadata to carry onto every resulting chunk.

        Returns:
            The chunks found in that page.
        """
        units = _split_into_units(page.text)
        if len(units) < 2:
            text = page.text.strip()
            if not text:
                return []
            return [self._build(page, text, headers, metadata)]

        vectors = self._embedder.embed_texts(units)
        distances = [
            1.0 - _cosine(vectors[index - 1], vectors[index])
            for index in range(1, len(vectors))
        ]
        threshold = _percentile(distances, self.breakpoint_percentile)

        texts = _group_units(
            units,
            distances,
            threshold=threshold,
            max_chars=self.max_chunk_chars,
            min_chars=self.min_chunk_chars,
        )
        return [self._build(page, text, headers, metadata) for text in texts]

    def _build(
        self,
        page: Page,
        text: str,
        headers: Sequence[str],
        metadata: dict[str, object] | None,
    ) -> Chunk:
        """Assemble a chunk with its provenance.

        Args:
            page: The page the text came from.
            text: The chunk's text.
            headers: The heading trail leading to the chunk.
            metadata: Metadata to carry onto the chunk.

        Returns:
            The assembled chunk.
        """
        trail = tuple(headers)
        return Chunk(
            document_id=page.document_id,
            page_number=page.page_number,
            text=text,
            section=SECTION_SEPARATOR.join(trail) if trail else None,
            headers=trail,
            ocr_extracted=page.ocr_extracted,
            metadata=dict(metadata or {}),
        )


def _split_into_units(text: str) -> list[str]:
    """Split text into the smallest pieces a boundary may fall between.

    Sentences are the natural unit, but table rows and headings are kept whole:
    a table row split from its neighbours is meaningless, and a heading belongs
    with the text beneath it.

    Args:
        text: The text to split.

    Returns:
        The units, in order, with blanks removed.
    """
    units: list[str] = []
    for block in text.split("\n"):
        stripped = block.strip()
        if not stripped:
            continue
        if _STRUCTURAL_LINE.match(block):
            units.append(stripped)
            continue
        units.extend(
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(stripped)
            if sentence.strip()
        )
    return units


def _group_units(
    units: Sequence[str],
    distances: Sequence[float],
    *,
    threshold: float,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    """Join units into chunks, breaking at large distances or the size limit.

    Args:
        units: The units to group, in order.
        distances: Distance between each unit and the one before it, so
            ``distances[i]`` precedes ``units[i + 1]``.
        threshold: Distance above which a boundary is placed.
        max_chars: Hard ceiling on a chunk's size.
        min_chars: Below this, a finished chunk is held open and merged with
            what follows rather than emitted alone.

    Returns:
        The grouped chunk texts.
    """
    chunks: list[str] = []
    current = [units[0]]
    length = len(units[0])

    for index in range(1, len(units)):
        unit = units[index]
        addition = len(unit) + 1
        topic_changed = distances[index - 1] > threshold
        too_long = length + addition > max_chars

        if (topic_changed and length >= min_chars) or too_long:
            chunks.append(" ".join(current))
            current = [unit]
            length = len(unit)
        else:
            current.append(unit)
            length += addition

    chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return the value at a percentile, by linear interpolation.

    Implemented here rather than pulled from numpy so this module keeps no
    numerical dependency of its own.

    Args:
        values: The observed values. Must not be empty.
        percentile: The percentile to take, between 0 and 100.

    Returns:
        The interpolated value at that percentile.
    """
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        left: The first vector.
        right: The second vector.

    Returns:
        Their cosine similarity, or 0.0 if either has zero magnitude.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_magnitude = sum(value * value for value in left) ** 0.5
    right_magnitude = sum(value * value for value in right) ** 0.5
    magnitude = left_magnitude * right_magnitude
    return dot / magnitude if magnitude else 0.0
