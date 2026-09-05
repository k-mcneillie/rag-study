"""Tests for the human feedback log."""

from __future__ import annotations

import json
from pathlib import Path

from rag.domain.models import (
    Answer,
    Chunk,
    PromptContext,
    PromptMetadata,
    RankedChunk,
    RerankedChunk,
)
from rag.feedback import FeedbackRecord, JsonlFeedbackLog

PASSAGE_TEXT = "The sampling frame excluded part-time staff."


def _context() -> PromptContext:
    """Build a context with one cited passage.

    Returns:
        A context carrying provenance and passage text.
    """
    chunk = Chunk(
        document_id="doc-1",
        page_number=12,
        text=PASSAGE_TEXT,
        section="2 Methods > 2.1 Sampling",
    )
    reranked = RerankedChunk(
        ranked_chunk=RankedChunk(chunk=chunk, score=0.7, rank=0),
        rerank_score=0.88,
        rank=0,
    )
    return PromptContext(
        chunks=(reranked,),
        rendered_text=f"...{PASSAGE_TEXT}...",
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )


def _answer() -> Answer:
    """Build a completed answer.

    Returns:
        An answer attributable to a model and a prompt.
    """
    return Answer(
        text="Part-time staff were excluded [1].",
        reasoning="Checking passage one.",
        model_name="test-model",
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )


def _read(path: Path) -> list[dict[str, object]]:
    """Read every record written to a log.

    Args:
        path: The log file.

    Returns:
        The parsed records, in the order they were written.
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_record_captures_the_turn_it_rates(tmp_path: Path) -> None:
    """The vote, question, answer, model, and prompt are all recorded.

    Args:
        tmp_path: Temporary directory for the log.
    """
    log = JsonlFeedbackLog(tmp_path / "feedback.jsonl")
    log.record(FeedbackRecord.build("up", "Who was excluded?", _answer(), _context()))

    (record,) = _read(log.path)
    assert record["vote"] == "up"
    assert record["query"] == "Who was excluded?"
    assert record["answer"] == "Part-time staff were excluded [1]."
    assert record["model_name"] == "test-model"
    assert record["prompt_version"] == "v1"
    assert record["recorded_at"].endswith("+00:00")


def test_citations_record_provenance_and_not_passage_text(tmp_path: Path) -> None:
    """A citation locates a passage; it never copies the corpus.

    The document pool is not something to duplicate into a second file on
    disk, and provenance identifies a passage precisely enough without it.

    Args:
        tmp_path: Temporary directory for the log.
    """
    log = JsonlFeedbackLog(tmp_path / "feedback.jsonl")
    log.record(FeedbackRecord.build("down", "Who?", _answer(), _context()))

    (record,) = _read(log.path)
    assert record["citations"] == [
        {
            "position": 1,
            "document_id": "doc-1",
            "page_number": 12,
            "section": "2 Methods > 2.1 Sampling",
            "score": 0.88,
        }
    ]
    assert PASSAGE_TEXT not in log.path.read_text(encoding="utf-8")


def test_records_are_appended_one_per_line(tmp_path: Path) -> None:
    """Writing never rewrites what is already there.

    Args:
        tmp_path: Temporary directory for the log.
    """
    log = JsonlFeedbackLog(tmp_path / "feedback.jsonl")
    for vote in ("up", "down", "up"):
        log.record(FeedbackRecord.build(vote, "Q?", _answer(), _context()))

    assert [record["vote"] for record in _read(log.path)] == ["up", "down", "up"]


def test_the_log_directory_is_created_on_first_write(tmp_path: Path) -> None:
    """A configured but unused log leaves nothing behind.

    Args:
        tmp_path: Temporary directory for the log.
    """
    log = JsonlFeedbackLog(tmp_path / "nested" / "dir" / "feedback.jsonl")
    assert not log.path.parent.exists()

    log.record(FeedbackRecord.build("up", "Q?", _answer(), _context()))

    assert log.path.exists()


def test_non_ascii_text_survives_the_round_trip(tmp_path: Path) -> None:
    """Feedback on documents that are not English is stored readably.

    Args:
        tmp_path: Temporary directory for the log.
    """
    log = JsonlFeedbackLog(tmp_path / "feedback.jsonl")
    log.record(FeedbackRecord.build("up", "Qué se midió?", _answer(), _context()))

    assert _read(log.path)[0]["query"] == "Qué se midió?"


def test_a_context_without_passages_records_no_citations(tmp_path: Path) -> None:
    """An answer given no passages is still ratable.

    Args:
        tmp_path: Temporary directory for the log.
    """
    empty = PromptContext(
        chunks=(),
        rendered_text="(no relevant source material was found)",
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )
    log = JsonlFeedbackLog(tmp_path / "feedback.jsonl")
    log.record(FeedbackRecord.build("down", "Q?", _answer(), empty))

    assert _read(log.path)[0]["citations"] == []
