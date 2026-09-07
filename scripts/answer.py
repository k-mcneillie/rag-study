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

from rag.assembly import build_chat_model, build_retrieval_orchestrator
from rag.config import ConfigurationError, load_settings
from rag.domain.models import PromptContext
from rag.generation import BaseChatModel, GenerationError
from rag.model_assets import ModelUnavailableError


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
        orchestrator = build_retrieval_orchestrator(settings)
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
