"""The FastAPI application factory.

``create_app`` builds the app; a ``lifespan`` handler composes the query
pipeline once, at startup, through :mod:`rag.assembly` — exactly as
``scripts/`` and ``app/`` do. A configuration problem or a missing model
raises here and the process exits non-zero: a service that started and then
failed on the first question would be harder to diagnose than one that never
started.

Run it with::

    uvicorn service.app:create_app --factory --host 127.0.0.1 --port 8080

The database must be reachable and the embedding model (and the reranker, if
one is configured) must be present. Nothing is downloaded.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.assembly import build_chat_model, build_retrieval_orchestrator
from rag.config import load_settings
from rag.feedback import JsonlFeedbackLog
from service.routes import router
from service.security import API_KEY_ENV

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline once and hold it for the app's lifetime.

    Args:
        app: The application whose state is being populated.

    Yields:
        Control back to the server once the pipeline is ready.
    """
    settings = load_settings()
    if not os.environ.get(API_KEY_ENV, "").strip():
        logger.warning(
            "%s is not set; the service is running OPEN — every route is "
            "reachable without a credential. Set it before exposing the service.",
            API_KEY_ENV,
        )

    app.state.settings = settings
    app.state.orchestrator = build_retrieval_orchestrator(settings)
    app.state.chat_model = build_chat_model(settings)
    app.state.feedback_log = JsonlFeedbackLog(settings.feedback_path)
    app.state.ingestion = None
    app.state.ingestion_lock = asyncio.Lock()

    logger.info(
        "Ready: model=%s provider=%s reranker=%s prompt=%s/%s",
        app.state.chat_model.model_name,
        settings.generation.provider,
        "cross-encoder" if settings.reranker_model_path else "vector-only",
        settings.prompt_name,
        settings.prompt_version,
    )
    yield


def create_app() -> FastAPI:
    """Build the API application.

    Returns:
        A FastAPI app with the four routes mounted and the startup pipeline
        build wired to its lifespan.
    """
    app = FastAPI(
        title="rag-study API",
        version="0.1.0",
        summary="Retrieval and answering over an offline document corpus.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
