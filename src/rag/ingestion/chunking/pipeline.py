"""The default chunking strategy, and the contamination filters it applies.

Structure first, then size: pages are cut at their Markdown headings so every
chunk knows which section it came from, and only those still over budget are
split further. Splitting by size alone would produce passages that cannot say
where they came from; splitting by structure alone would leave whole sections
in a single chunk.

Two filters run afterwards, both addressing contamination that degrades
retrieval rather than anything wrong with the splitting itself:

* **Reference lists.** A paper's bibliography is a long run of author names,
  titles, and venues. It matches queries on wording alone while answering
  nothing, and in a long paper it can outweigh the content. Chunks under a
  references heading are dropped by default and can be kept by configuration.
* **Fragments.** Chunks consisting of a bare heading or a stray line embed
  poorly and displace real passages from the top-k.

Both filters run after chunking rather than during cleaning, because both need
to know a chunk's section, which only exists once the document has been split.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.domain.models import Chunk, Page
from rag.ingestion.chunking.splitters import MarkdownHeaderChunker, RecursiveChunker
from rag.ingestion.interfaces import BaseChunker

#: Headings that open a reference list. Matched against the heading trail, so a
#: subsection of the bibliography is caught along with the section itself.
_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|works\s+cited|literature\s+cited)"
    r"\s*$",
    re.IGNORECASE,
)

#: Key under which a chunk records that it came from a reference list.
SECTION_TYPE_KEY = "section_type"

#: Value stored under :data:`SECTION_TYPE_KEY` for reference-list chunks.
REFERENCES_SECTION = "references"


class ChunkingPipeline(BaseChunker):
    """Chunks by structure, then by size, then drops known contamination.

    Attributes:
        min_chunk_chars: Shortest chunk worth keeping.
        drop_references: Whether to discard reference-list chunks.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        min_chunk_chars: int = 40,
        drop_references: bool = True,
    ) -> None:
        """Initialise the pipeline.

        Args:
            chunk_size: Target maximum chunk size, in characters.
            chunk_overlap: Characters shared between adjacent chunks.
            min_chunk_chars: Shortest chunk worth keeping. Guards against
                chunks that are just a heading or a stray fragment.
            drop_references: Whether to discard chunks from reference lists.
                Set to ``False`` to retain them, for example to answer
                questions about what a paper cites.

        Raises:
            ValueError: If ``min_chunk_chars`` is negative, or the chunk sizes
                are invalid.
        """
        if min_chunk_chars < 0:
            raise ValueError("min_chunk_chars must be non-negative.")

        self.min_chunk_chars = min_chunk_chars
        self.drop_references = drop_references
        self._structural = MarkdownHeaderChunker()
        self._recursive = RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def chunk(self, pages: Sequence[Page]) -> list[Chunk]:
        """Chunk pages and remove contamination.

        Args:
            pages: The pages to split, in source order.

        Returns:
            The retained chunks, in document order.
        """
        sections = self._structural.chunk(pages)

        sized: list[Chunk] = []
        for section in sections:
            sized.extend(self._recursive.split_oversized(section))

        return [
            tagged
            for chunk in sized
            if (tagged := self._tag(chunk)) is not None and self._keep(tagged)
        ]

    def _tag(self, chunk: Chunk) -> Chunk:
        """Record whether a chunk came from a reference list.

        Args:
            chunk: The chunk to inspect.

        Returns:
            The chunk, with its section type recorded when it is part of a
            reference list.
        """
        if not _is_reference_section(chunk.headers):
            return chunk
        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            text=chunk.text,
            section=chunk.section,
            headers=chunk.headers,
            metadata={**chunk.metadata, SECTION_TYPE_KEY: REFERENCES_SECTION},
        )

    def _keep(self, chunk: Chunk) -> bool:
        """Decide whether a chunk survives the contamination filters.

        Args:
            chunk: The chunk to test.

        Returns:
            ``True`` if the chunk should be retained.
        """
        if len(chunk.text) < self.min_chunk_chars:
            return False
        is_references = chunk.metadata.get(SECTION_TYPE_KEY) == REFERENCES_SECTION
        return not (self.drop_references and is_references)


def _is_reference_section(headers: Sequence[str]) -> bool:
    """Report whether a heading trail sits inside a reference list.

    Args:
        headers: The heading trail leading to a chunk.

    Returns:
        ``True`` if any heading in the trail opens a reference list.
    """
    return any(_REFERENCE_HEADING.match(header) for header in headers)
