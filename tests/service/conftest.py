"""Fixtures and fakes for the API tests.

The API is exercised with a hand-built app: a bare :class:`fastapi.FastAPI`
with the real router mounted and ``app.state`` populated with fakes. The
lifespan handler in :mod:`service.app` — which would load models and open a
database — is deliberately not run, so these tests need no weights, no
database, and no network.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.domain.models import (
    AnswerDelta,
    Chunk,
    PromptContext,
    PromptMetadata,
    RankedChunk,
    RerankedChunk,
)
from rag.feedback import JsonlFeedbackLog
from rag.generation import GenerationError
from rag.ingestion import IngestionResult

API_KEY = "test-key"


def make_context(*, chunks: int = 2) -> PromptContext:
    """Build a small assembled context.

    Args:
        chunks: How many reranked chunks to include.

    Returns:
        A context whose chunks carry traceable provenance.
    """
    reranked = tuple(
        RerankedChunk(
            ranked_chunk=RankedChunk(
                chunk=Chunk(
                    id=f"chunk-{i}",
                    document_id="doc-1",
                    page_number=i + 1,
                    text=f"Passage {i}.",
                    section=f"{i + 1} Section",
                ),
                score=0.9 - i / 10,
                rank=i,
            ),
            rerank_score=0.8 - i / 10,
            rank=i,
        )
        for i in range(chunks)
    )
    return PromptContext(
        chunks=reranked,
        rendered_text="RENDERED PROMPT",
        prompt_metadata=PromptMetadata(
            prompt_name="retrieval_context", prompt_version="v1"
        ),
    )


@dataclass
class FakeOrchestrator:
    """A retrieval orchestrator returning a canned context.

    Attributes:
        context: What :meth:`retrieve` returns.
        error: If set, :meth:`retrieve` raises ``ValueError`` with this text.
        calls: The ``(query, top_k)`` pairs it was called with.
    """

    context: PromptContext = field(default_factory=make_context)
    error: str | None = None
    calls: list[tuple[str, int]] = field(default_factory=list)

    def retrieve(self, query: str, *, top_k: int = 5) -> PromptContext:
        """Return the canned context, recording the call.

        Args:
            query: The query.
            top_k: Requested passage count.

        Returns:
            The canned context.

        Raises:
            ValueError: If ``error`` is set.
        """
        self.calls.append((query, top_k))
        if self.error is not None:
            raise ValueError(self.error)
        return self.context


@dataclass
class FakeChatModel:
    """A chat model yielding canned fragments.

    Attributes:
        deltas: Fragments to yield from :meth:`stream`.
        error: If set, :meth:`stream` raises ``GenerationError`` with this
            text after yielding nothing.
        model_name: The reported model identity.
    """

    deltas: Sequence[AnswerDelta] = field(
        default_factory=lambda: (
            AnswerDelta(text="Thinking. ", reasoning=True),
            AnswerDelta(text="The answer is 42 [1]."),
        )
    )
    error: str | None = None
    model_name: str = "fake-model"

    def stream(self, context: PromptContext) -> Iterator[AnswerDelta]:
        """Yield the canned fragments.

        Args:
            context: Ignored.

        Yields:
            The configured fragments.

        Raises:
            GenerationError: If ``error`` is set.
        """
        del context
        if self.error is not None:
            raise GenerationError(self.error)
        yield from self.deltas


@dataclass
class FakeIngestion:
    """An ingestion orchestrator returning a canned result.

    Attributes:
        result: What :meth:`ingest` returns.
        calls: The paths it was asked to ingest.
    """

    result: IngestionResult = field(
        default_factory=lambda: IngestionResult(
            source=Path("upload.pdf"), document_id="doc-9", chunk_count=3
        )
    )
    calls: list[Path] = field(default_factory=list)

    def ingest(self, source: Path, *, skip_duplicates: bool = True) -> IngestionResult:
        """Return the canned result, recording the call.

        Args:
            source: The uploaded file.
            skip_duplicates: Ignored.

        Returns:
            The configured result.
        """
        del skip_duplicates
        self.calls.append(source)
        return self.result


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the shared secret the routes check against.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    monkeypatch.setenv("RAG_API_KEY", API_KEY)


@dataclass
class Harness:
    """A test app and the fakes wired into it.

    Attributes:
        client: A test client for the app.
        orchestrator: The retrieval fake on ``app.state``.
        chat_model: The generation fake on ``app.state``.
        ingestion: The ingestion fake on ``app.state``.
        feedback_path: Where ratings are written.
    """

    client: TestClient
    orchestrator: FakeOrchestrator
    chat_model: FakeChatModel
    ingestion: FakeIngestion
    feedback_path: Path
    settings: object


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Build an app with fake collaborators and a real feedback log.

    Args:
        tmp_path: A temporary directory for the feedback file.

    Yields:
        The harness.
    """
    from service.routes import router

    orchestrator = FakeOrchestrator()
    chat_model = FakeChatModel()
    ingestion = FakeIngestion()
    feedback_path = tmp_path / "feedback.jsonl"

    class _Settings:
        top_k = 5
        max_document_bytes = 1_000_000
        prompt_name = "retrieval_context"
        prompt_version = "v1"
        reranker_model_path = None

        class generation:  # noqa: N801 - mirrors the real nested settings object
            provider = "ollama"

    settings = _Settings()
    app = FastAPI()
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    app.state.chat_model = chat_model
    app.state.ingestion = ingestion
    app.state.feedback_log = JsonlFeedbackLog(feedback_path)
    app.include_router(router)

    with TestClient(app) as client:
        yield Harness(
            client=client,
            orchestrator=orchestrator,
            chat_model=chat_model,
            ingestion=ingestion,
            feedback_path=feedback_path,
            settings=settings,
        )


def parse_sse(body: str) -> list[tuple[str, str]]:
    """Split a server-sent-events body into ``(event, data)`` pairs.

    Args:
        body: The raw response text.

    Returns:
        One pair per event block, in order. Blocks that are only comments
        (keep-alive pings) are skipped.
    """
    events: list[tuple[str, str]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if name:
            events.append((name, "\n".join(data_lines)))
    return events
