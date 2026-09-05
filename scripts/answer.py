"""Retrieve context and answer from it. The full loop, without a browser.

This is ``query.py`` with a model on the end. It exists mostly to make the
generation step debuggable on its own: the same retrieval, the same prompt,
and the answer printed as it streams, with none of the interface in the way.

The model service must already be running and must already have the model.
Nothing is downloaded here.

Usage::

    python scripts/answer.py "How does DPO avoid training a reward model?"
    python scripts/answer.py "..." --top-k 3 --show-reasoning --show-prompt
"""

from __future__ import annotations

import argparse
import sys

from rag.config import ConfigurationError, Settings, load_settings
from rag.domain.models import PromptContext
from rag.generation import BaseChatModel, GenerationError, chat_model_for
from rag.model_assets import ModelUnavailableError
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
        ModelUnavailableError: If a reranker is configured but missing.
    """
    if settings.reranker_model_path is None:
        return PassthroughReranker()
    return CrossEncoderReranker(
        settings.reranker_model_path, batch_size=settings.reranker_batch_size
    )


def build_orchestrator(settings: Settings) -> RetrievalOrchestrator:
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


def main(argv: list[str] | None = None) -> int:
    """Answer a question from the corpus and print its citations.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv``.

    Returns:
        A process exit status: ``0`` on success, ``1`` if configuration, a
        model, or the model service is unavailable, or the query is rejected.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="The question to answer.")
    parser.add_argument("--top-k", type=int, help="Passages to retrieve.")
    parser.add_argument(
        "--document-id", help="Restrict retrieval to a single document."
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print the rendered prompt."
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print the model's reasoning as well as its answer.",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        orchestrator = build_orchestrator(settings)
        chat_model = build_chat_model(settings)
    except (ConfigurationError, ModelUnavailableError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1

    filters = {"document_id": args.document_id} if args.document_id else None
    top_k = args.top_k if args.top_k is not None else settings.top_k
    try:
        context = orchestrator.retrieve(args.query, top_k=top_k, filters=filters)
    except ValueError as exc:
        print(f"Query rejected: {exc}", file=sys.stderr)
        return 1

    if args.show_prompt:
        print(context.rendered_text)
        print("-" * 72)

    if not context.chunks:
        print("No relevant passages were found; the model was not called.")
        return 0

    if _stream(chat_model, context, show_reasoning=args.show_reasoning) != 0:
        return 1

    print("\n\nSources")
    for position, chunk in enumerate(context.chunks, start=1):
        source = chunk.chunk
        print(
            f"  [{position}] score={chunk.rerank_score:.3f}  "
            f"page {source.page_number}  {source.section or '-'}  "
            f"({source.document_id})"
        )
    return 0


def _stream(
    chat_model: BaseChatModel, context: PromptContext, *, show_reasoning: bool
) -> int:
    """Print an answer as it arrives.

    Args:
        chat_model: The model to answer with.
        context: The assembled retrieval context.
        show_reasoning: Whether to print reasoning alongside the answer.

    Returns:
        ``0`` when the answer completed, ``1`` if generation failed.
    """
    reasoning_shown = False
    try:
        for delta in chat_model.stream(context):
            if delta.reasoning:
                if not show_reasoning:
                    continue
                if not reasoning_shown:
                    print("[reasoning] ", end="", flush=True)
                    reasoning_shown = True
            elif reasoning_shown:
                print("\n\n", end="", flush=True)
                reasoning_shown = False
            print(delta.text, end="", flush=True)
    except GenerationError as exc:
        print(f"\nGeneration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
