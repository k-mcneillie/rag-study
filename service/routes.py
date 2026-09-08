"""The HTTP routes: health, answer, feedback, documents.

Every route reads its collaborators from ``request.app.state``, which
:func:`service.app.create_app` populates once at startup. Blocking pipeline
calls are pushed to a worker thread with
:func:`starlette.concurrency.run_in_threadpool`, so a slow retrieval or a slow
ingest does not block the event loop.

When ``RAG_API_KEY`` is set, every route except ``GET /health`` requires it;
see :func:`service.security.require_api_key`.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from rag.feedback import Citation, FeedbackRecord, JsonlFeedbackLog
from rag.ingestion import IngestionOrchestrator
from service.ingest_wiring import build_ingestion_orchestrator
from service.schemas import (
    AnswerRequest,
    DocumentOut,
    FeedbackRequest,
    HealthOut,
)
from service.security import require_api_key
from service.streaming import answer_events, no_context_event

logger = logging.getLogger(__name__)

router = APIRouter()

#: Applied to every route that touches data. ``/health`` is left open so a
#: load balancer or a person can check the service without a credential.
_KEYED = [Depends(require_api_key)]


@router.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Report what the service is configured to do. No authentication.

    Args:
        request: The incoming request, carrying the app state.

    Returns:
        The configured model, provider, reranker, prompt, and default top-k.
    """
    settings = request.app.state.settings
    chat_model = request.app.state.chat_model
    return HealthOut(
        model=chat_model.model_name,
        provider=settings.generation.provider,
        reranker="cross-encoder" if settings.reranker_model_path else "vector-only",
        prompt=f"{settings.prompt_name}/{settings.prompt_version}",
        top_k=settings.top_k,
    )


async def _single_event(event: dict[str, str]) -> AsyncIterator[dict[str, str]]:
    """Wrap one SSE event as an async stream of length one.

    Args:
        event: The event to emit.

    Yields:
        The event, once.
    """
    yield event


@router.post("/answer", dependencies=_KEYED)
async def answer(request: Request, body: AnswerRequest) -> EventSourceResponse:
    """Retrieve context and stream an answer as server-sent events.

    Args:
        request: The incoming request, carrying the app state.
        body: The question and an optional passage count.

    Returns:
        A ``text/event-stream`` response. See :mod:`service.streaming` for the
        event grammar. When retrieval matches nothing, the stream is a single
        ``no_context`` event and the model is never called.

    Raises:
        HTTPException: 422 if the query is empty or the passage count is out of
            range.
    """
    settings = request.app.state.settings
    orchestrator = request.app.state.orchestrator
    chat_model = request.app.state.chat_model

    top_k = body.top_k if body.top_k is not None else settings.top_k
    try:
        context = await run_in_threadpool(
            orchestrator.retrieve, body.query, top_k=top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not context.chunks:
        return EventSourceResponse(_single_event(no_context_event()))
    return EventSourceResponse(answer_events(chat_model, context))


@router.post("/feedback", status_code=204, dependencies=_KEYED)
async def feedback(request: Request, body: FeedbackRequest) -> Response:
    """Append one rating to the feedback log.

    Args:
        request: The incoming request, carrying the app state.
        body: The rating and the fields the log records.

    Returns:
        An empty 204 response.

    Raises:
        HTTPException: 503 if the feedback log cannot be written.
    """
    log: JsonlFeedbackLog = request.app.state.feedback_log
    record = FeedbackRecord(
        vote=body.vote,
        query=body.query,
        answer=body.answer,
        model_name=body.model_name,
        prompt_name=body.prompt.name,
        prompt_version=body.prompt.version,
        citations=tuple(
            Citation(
                position=item.position,
                document_id=item.document_id,
                page_number=item.page_number,
                section=item.section,
                score=item.score,
            )
            for item in body.citations
        ),
    )
    try:
        await run_in_threadpool(log.record, record)
    except OSError as exc:
        logger.exception("Could not write feedback to %s.", log.path)
        raise HTTPException(
            status_code=503, detail="The feedback log is unwritable."
        ) from exc
    return Response(status_code=204)


async def _ingestion_orchestrator(request: Request) -> IngestionOrchestrator:
    """Return the ingestion pipeline, building it once on first use.

    The embedding model it loads is not needed to answer questions, so it is
    not built at startup. Concurrent first calls are serialised so the model
    is loaded once.

    Args:
        request: The incoming request, carrying the app state.

    Returns:
        The wired :class:`~rag.ingestion.IngestionOrchestrator`.
    """
    state = request.app.state
    if state.ingestion is None:
        async with state.ingestion_lock:
            if state.ingestion is None:
                state.ingestion = await run_in_threadpool(
                    build_ingestion_orchestrator, state.settings
                )
    orchestrator: IngestionOrchestrator = state.ingestion
    return orchestrator


@router.post("/documents", dependencies=_KEYED)
async def documents(request: Request, file: UploadFile) -> Response:
    """Ingest one uploaded PDF, blocking until it is indexed.

    Args:
        request: The incoming request, carrying the app state.
        file: The uploaded document. Must be a PDF within the configured size
            limit.

    Returns:
        201 with the new document id and chunk count when stored; 200 with
        ``already_indexed`` when identical content was already present.

    Raises:
        HTTPException: 415 for a non-PDF, 413 when the file exceeds
            ``RAG_MAX_DOCUMENT_BYTES``, 422 when the document yields no usable
            text.
    """
    settings = request.app.state.settings
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    payload = await file.read()
    if len(payload) > settings.max_document_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_document_bytes}-byte limit.",
        )

    orchestrator = await _ingestion_orchestrator(request)
    tmp_path = Path(
        tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name  # noqa: SIM115
    )
    try:
        tmp_path.write_bytes(payload)
        result = await run_in_threadpool(
            orchestrator.ingest, tmp_path, skip_duplicates=True
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.error:
        raise HTTPException(status_code=422, detail=result.error)
    if result.skipped:
        return JSONResponse(
            status_code=200, content=DocumentOut(status="already_indexed").model_dump()
        )
    return JSONResponse(
        status_code=201,
        content=DocumentOut(
            status="ingested",
            document_id=result.document_id,
            chunk_count=result.chunk_count,
        ).model_dump(),
    )
