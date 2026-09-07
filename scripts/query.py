"""Query the store. The retrieval pipeline, end to end.

Like its ingestion counterpart, this script only assembles components. It runs
without the ingestion package being importable at all, which is the practical
demonstration that the two pipelines are independent.

The reranker is used when one is configured and skipped when it is not, so the
same script shows both arrangements.

Usage::

    python scripts/query.py "How does DPO avoid training a reward model?"
    python scripts/query.py "..." --top-k 3 --show-prompt

No language model is called. The output is the assembled context and its
provenance, which is where this package stops.
"""

from __future__ import annotations

import argparse
import sys

from rag.config import ConfigurationError, Settings, load_settings
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


def main(argv: list[str] | None = None) -> int:
    """Retrieve context for a query and print it with its provenance.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv``.

    Returns:
        A process exit status: ``0`` on success, ``1`` if configuration or a
        model is missing, or the query is rejected.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=5, help="Passages to return.")
    parser.add_argument(
        "--document-id", help="Restrict retrieval to a single document."
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print the rendered prompt."
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        orchestrator = build_orchestrator(settings)
    except (ConfigurationError, ModelUnavailableError) as exc:
        print(f"Cannot start: {exc}", file=sys.stderr)
        return 1

    filters = {"document_id": args.document_id} if args.document_id else None
    try:
        context = orchestrator.retrieve(args.query, top_k=args.top_k, filters=filters)
    except ValueError as exc:
        print(f"Query rejected: {exc}", file=sys.stderr)
        return 1

    reranked = "cross-encoder" if settings.reranker_model_path else "none (vector only)"
    print(
        f"prompt: {context.prompt_metadata.prompt_name} "
        f"{context.prompt_metadata.prompt_version}   reranker: {reranked}\n"
    )
    for position, chunk in enumerate(context.chunks, start=1):
        source = chunk.chunk
        print(
            f"[{position}] score={chunk.rerank_score:.3f} "
            f"(initial {chunk.initial_score:.3f})  "
            f"page {source.page_number}  {source.section or '-'}"
        )
        print(f"    {source.text[:200].strip()}\n")

    if not context.chunks:
        print("No relevant passages were found.")

    if args.show_prompt:
        print("-" * 72)
        print(context.rendered_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
