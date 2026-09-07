"""Recording human judgements about answers.

Retrieval quality here is, as the documentation admits, unmeasured. A rating
attached to a real question, with the passages that were actually retrieved to
answer it, is the cheapest evidence available: it costs the reader one click
and accumulates into the sample an evaluation harness will eventually need.

The log is a JSON Lines file rather than a table. Feedback is append-only, is
read far less often than it is written, and should survive the database being
unavailable — and a line-oriented file means an interrupted write costs at
most the record being written.

What is recorded is deliberately narrow. Passage text is never written: the
citation provenance identifies a passage precisely, and the corpus this system
reads is not something to copy into a second place on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag.domain.models import Answer, PromptContext

#: The permitted votes. Free text belongs in a comment field, not here.
Vote = Literal["up", "down"]


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC timestamp.

    Returns:
        The current UTC time.
    """
    return datetime.now(UTC)


@dataclass(frozen=True)
class Citation:
    """Where one cited passage came from.

    Attributes:
        position: The passage's one-based number in the prompt, matching the
            ``[n]`` marker the model was asked to cite.
        document_id: Identifier of the document the passage belongs to.
        page_number: Page the passage came from.
        section: Heading path within the document, if one was detected.
        score: The reranker's score for the passage.
    """

    position: int
    document_id: str
    page_number: int
    section: str | None
    score: float


@dataclass(frozen=True)
class FeedbackRecord:
    """One rating of one answer.

    Attributes:
        vote: Whether the answer was judged helpful.
        query: The question that was asked.
        answer: The answer that was rated.
        model_name: Which model produced the answer.
        prompt_name: Logical name of the prompt template used.
        prompt_version: Version of that template.
        citations: Provenance of the passages the answer was given, never
            their text.
        recorded_at: When the rating was given.
    """

    vote: Vote
    query: str
    answer: str
    model_name: str
    prompt_name: str
    prompt_version: str
    citations: tuple[Citation, ...] = ()
    recorded_at: datetime = field(default_factory=_utc_now)

    @classmethod
    def build(
        cls, vote: Vote, query: str, answer: Answer, context: PromptContext
    ) -> FeedbackRecord:
        """Assemble a record from a completed turn.

        Args:
            vote: The rating given.
            query: The question that was asked.
            answer: The answer that was produced.
            context: The context the answer was produced from.

        Returns:
            The record to write.
        """
        return cls(
            vote=vote,
            query=query,
            answer=answer.text,
            model_name=answer.model_name,
            prompt_name=context.prompt_metadata.prompt_name,
            prompt_version=context.prompt_metadata.prompt_version,
            citations=tuple(
                Citation(
                    position=position,
                    document_id=chunk.chunk.document_id,
                    page_number=chunk.chunk.page_number,
                    section=chunk.chunk.section,
                    score=chunk.rerank_score,
                )
                for position, chunk in enumerate(context.chunks, start=1)
            ),
        )

    def as_json(self) -> dict[str, Any]:
        """Render the record as a JSON-compatible mapping.

        Returns:
            The record's fields, with the timestamp in ISO 8601 form.
        """
        return {
            "recorded_at": self.recorded_at.isoformat(),
            "vote": self.vote,
            "query": self.query,
            "answer": self.answer,
            "model_name": self.model_name,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "citations": [
                {
                    "position": citation.position,
                    "document_id": citation.document_id,
                    "page_number": citation.page_number,
                    "section": citation.section,
                    "score": citation.score,
                }
                for citation in self.citations
            ],
        }


class JsonlFeedbackLog:
    """Appends feedback records to a JSON Lines file."""

    def __init__(self, path: Path) -> None:
        """Prepare a log at a given path.

        The file is not created until the first record is written, so a
        configured but unused log leaves nothing behind.

        Args:
            path: File to append to. Its parent directory is created on first
                write if it does not exist.
        """
        self._path = path

    @property
    def path(self) -> Path:
        """Return the file being written to.

        Returns:
            The log's path.
        """
        return self._path

    def record(self, feedback: FeedbackRecord) -> None:
        """Append one record to the log.

        Args:
            feedback: The record to write.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(feedback.as_json(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
