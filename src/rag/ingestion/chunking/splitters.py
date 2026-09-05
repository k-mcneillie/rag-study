"""Structure-aware and size-aware chunking.

Both chunkers here are thin adapters over ``langchain_text_splitters``. Where a
Markdown heading starts a new section, and how to back off through paragraph,
line, sentence, and word boundaries when a passage runs long, are well-solved
problems; using a maintained implementation keeps those edge cases someone
else's to maintain.

The adapters exist to keep that library an implementation detail. Each
subclasses :class:`~rag.ingestion.interfaces.BaseChunker` and returns domain
:class:`~rag.domain.models.Chunk` objects carrying their own provenance, so the
splitter underneath can be replaced without any other stage noticing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.domain.models import Chunk, Page
from rag.ingestion.interfaces import BaseChunker

#: Heading levels to split on, paired with the metadata key each populates,
#: ordered so the recovered trail reads from the outermost section inwards.
_HEADERS_TO_SPLIT_ON = [("#" * level, f"h{level}") for level in range(1, 7)]

#: Separator between levels of the section path.
SECTION_SEPARATOR = " > "

#: Back-off order used when a chunk must be split by size. Paragraph breaks
#: come first, so a Markdown table — which contains no blank lines — is only
#: broken into when a single table exceeds the budget on its own.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

#: Markdown emphasis and inline-code markers. Headings arrive with their
#: formatting intact ("**References**"), which would otherwise leak into the
#: section path and defeat anything matching on the heading's wording.
_EMPHASIS = re.compile(r"[*_`]+")


def normalise_heading(heading: str) -> str:
    """Reduce a heading to its plain wording.

    Args:
        heading: The heading text as captured from the document.

    Returns:
        The heading without Markdown emphasis or surrounding whitespace.
    """
    return _EMPHASIS.sub("", heading).strip()


class MarkdownHeaderChunker(BaseChunker):
    """Splits pages at Markdown headings, recording the heading trail.

    A section that runs across a page break keeps its heading trail on the
    following pages. Splitting page by page preserves exact page provenance,
    but on its own it would leave the continuation of a section looking like
    untitled text, so the trail in force at the end of one page carries into
    the next.
    """

    def __init__(self) -> None:
        """Initialise the chunker."""
        # Headings stay in the chunk text: they are part of the passage a
        # reader wants to see, and they help the embedding capture the
        # section's subject.
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Split pages into one chunk per heading section.

        Text appearing before a document's first heading becomes its own
        chunk, so abstracts and front matter are never discarded.

        Args:
            pages: The pages to split, in source order.

        Returns:
            The resulting chunks, in document order.
        """
        chunks: list[Chunk] = []
        # Heading in force at each level, carried across page boundaries.
        carried: dict[int, str] = {}

        for page in pages:
            for section in self._splitter.split_text(page.text):
                text = section.page_content.strip()
                if not text:
                    continue
                carried = _merge_headings(carried, _own_headings(section.metadata))
                trail = tuple(carried[level] for level in sorted(carried))
                chunks.append(build_chunk(page, text, trail))

        return chunks


class RecursiveChunker(BaseChunker):
    """Splits text down to a size budget, preserving provenance.

    Attributes:
        chunk_size: Target maximum size of a chunk, in characters.
        chunk_overlap: Characters repeated between neighbouring chunks, so a
            passage split mid-argument still carries its lead-in.
    """

    def __init__(self, *, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Target maximum chunk size, in characters.
            chunk_overlap: Characters shared between adjacent chunks.

        Raises:
            ValueError: If the sizes are not positive, or the overlap is not
                smaller than the chunk size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and below chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_RECURSIVE_SEPARATORS,
        )

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Split pages into size-bounded chunks.

        Args:
            pages: The pages to split, in source order.

        Returns:
            The resulting chunks, in document order.
        """
        chunks: list[Chunk] = []
        for page in pages:
            for text in self._splitter.split_text(page.text):
                if text.strip():
                    chunks.append(build_chunk(page, text.strip(), ()))
        return chunks

    def split_oversized(self, chunk: Chunk) -> list[Chunk]:
        """Split one chunk if it exceeds the budget, carrying its provenance.

        Args:
            chunk: The chunk to split.

        Returns:
            The chunk unchanged if it is within budget, otherwise its pieces,
            each retaining the original's page and section provenance.
        """
        if len(chunk.text) <= self.chunk_size:
            return [chunk]

        return [
            Chunk(
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                text=piece.strip(),
                section=chunk.section,
                headers=chunk.headers,
                metadata=dict(chunk.metadata),
            )
            for piece in self._splitter.split_text(chunk.text)
            if piece.strip()
        ]


def _own_headings(metadata: dict[str, object]) -> dict[int, str]:
    """Extract the headings a section declares for itself.

    Args:
        metadata: Section metadata from the splitter, keyed ``h1`` to ``h6``.

    Returns:
        The section's own headings, keyed by level.
    """
    headings = {}
    for level, (_, key) in enumerate(_HEADERS_TO_SPLIT_ON, start=1):
        value = metadata.get(key)
        if value and (text := normalise_heading(str(value))):
            headings[level] = text
    return headings


def _merge_headings(carried: dict[int, str], own: dict[int, str]) -> dict[int, str]:
    """Combine a section's own headings with those still in force.

    A page beginning part-way down a document's hierarchy reports only the
    heading levels that appear on that page, so a subsection continuing across
    a page break would otherwise lose the sections containing it. Levels above
    the shallowest heading the section declares are inherited; levels at or
    below it are replaced, because declaring a heading closes everything nested
    under the previous one.

    Args:
        carried: Heading in force at each level before this section.
        own: Headings this section declares, keyed by level.

    Returns:
        The heading in force at each level for this section.
    """
    if not own:
        return carried
    shallowest = min(own)
    ancestors = {level: text for level, text in carried.items() if level < shallowest}
    return ancestors | own


def build_chunk(page: Page, text: str, headers: Sequence[str]) -> Chunk:
    """Assemble a chunk carrying its page and section provenance.

    Args:
        page: The page the text came from.
        text: The chunk's text.
        headers: The heading trail leading to the chunk.

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
        metadata={"ocr_extracted": page.ocr_extracted},
    )
