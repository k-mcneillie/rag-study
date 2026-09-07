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

from rag.assembly import build_retrieval_orchestrator
from rag.config import ConfigurationError, load_settings
from rag.model_assets import ModelUnavailableError


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
        orchestrator = build_retrieval_orchestrator(settings)
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
