"""Compose the retrieval pipeline and its chat model from settings.

Every entry point that answers or retrieves queries — the two scripts and the
chat app — needs to build the same handful of collaborators from the same
``Settings``. This module holds that composition once, so a change to how the
pipeline is wired is made in one place instead of three.

It sits above both pipelines rather than inside either: it depends on
``retrieval``, ``storage``, and ``generation``, but never on ``ingestion``, so
it does not weaken the boundary between the two pipelines.
"""

from __future__ import annotations

from rag.config import Settings
from rag.generation import BaseChatModel, chat_model_for
from rag.retrieval import (
    CosineSimilarityRanker,
    CrossEncoderReranker,
    PassthroughReranker,
    RetrievalOrchestrator,
    TemplatePromptAugmenter,
)
from rag.retrieval.embedding.local import LocalSentenceTransformerQueryEmbedder
from rag.retrieval.interfaces import BaseReranker
from rag.storage import MariaDBRepository, build_engine, build_session_factory


def build_reranker(settings: Settings) -> BaseReranker:
    """Choose a reranker based on what has been provisioned.

    Args:
        settings: Application settings.

    Returns:
        A cross-encoder reranker when one is configured, otherwise the
        passthrough, which keeps the initial ranking.

    Raises:
        ModelUnavailableError: If a reranker is configured but missing. A
            configured model that cannot be loaded is an error rather than a
            silent downgrade, so a misconfiguration is never mistaken for a
            deliberate choice to skip reranking.
    """
    if settings.reranker_model_path is None:
        return PassthroughReranker()
    return CrossEncoderReranker(
        settings.reranker_model_path, batch_size=settings.reranker_batch_size
    )


def build_retrieval_orchestrator(settings: Settings) -> RetrievalOrchestrator:
    """Assemble the retrieval pipeline from configuration.

    Args:
        settings: Application settings.

    Returns:
        The wired orchestrator.

    Raises:
        ModelUnavailableError: If a configured model is not provisioned.
    """
    sessions = build_session_factory(build_engine(settings.database))
    repository = MariaDBRepository(
        sessions,
        embedding_dimension=settings.embedding_dimension,
        max_top_k=settings.max_top_k,
    )
    return RetrievalOrchestrator(
        query_embedder=LocalSentenceTransformerQueryEmbedder(
            settings.embedding_model_path
        ),
        ranker=CosineSimilarityRanker(repository),
        reranker=build_reranker(settings),
        prompt_augmenter=TemplatePromptAugmenter(
            prompt_name=settings.prompt_name, prompt_version=settings.prompt_version
        ),
        max_top_k=settings.max_top_k,
    )


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build the client for the answering model.

    Args:
        settings: Application settings.

    Returns:
        A client for whichever service ``RAG_LLM_PROVIDER`` names. Pointing
        this at a different service is an environment change, not a code
        change; adding a service it cannot yet speak to means one new
        ``BaseChatModel`` and one line in ``rag.generation.providers``.

    Raises:
        GenerationError: If the configured provider is not implemented.
    """
    return chat_model_for(settings.generation)
