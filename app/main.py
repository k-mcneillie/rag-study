"""A chat interface over the retrieval pipeline.

This is a launcher, not a layer. It lives outside ``src/rag`` for the same
reason the scripts do: the package is a library that assembles differently in
different contexts, and the assembly belongs to the context. Deleting this
directory removes the interface and nothing else.

It also has to bridge two worlds. Chainlit is asynchronous; the package is
deliberately not. Rather than colour every signature in the package with
``async``, the blocking work is pushed into a worker thread here, at the one
boundary that actually needs it.

Run it with::

    chainlit run app/main.py

The database must be reachable and the model service must already be running
with the configured model available. Neither is started from here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import AsyncExitStack
from typing import Any

import chainlit as cl

from rag.assembly import build_chat_model, build_retrieval_orchestrator
from rag.config import ConfigurationError, Settings, load_settings
from rag.domain.models import Answer, AnswerDelta, PromptContext
from rag.feedback import FeedbackRecord, JsonlFeedbackLog, Vote
from rag.generation import BaseChatModel, GenerationError
from rag.model_assets import ModelUnavailableError
from rag.retrieval import RetrievalOrchestrator

logger = logging.getLogger(__name__)

#: Session keys. Named once so a typo cannot silently read back ``None``.
_SETTINGS = "settings"
_ORCHESTRATOR = "orchestrator"
_CHAT_MODEL = "chat_model"
_FEEDBACK_LOG = "feedback_log"
_TURNS = "turns"


class _Turn:
    """One answered question, held until it is rated or the session ends.

    A rating arrives long after the answer was streamed, as a click carrying
    nothing but a message id. Everything the record needs is kept here so the
    rating never has to be reconstructed from what is on screen.

    This is written out by hand rather than as a dataclass, and must stay that
    way. Chainlit executes this file without registering it in
    ``sys.modules``, and ``dataclasses`` resolves the string annotations that
    ``from __future__ import annotations`` produces by looking the defining
    module up there. It is not found, and the decorator raises at import time.

    Attributes:
        message: The answer message, so its buttons can be retired.
        query: The question that was asked.
        answer: The answer that was produced.
        context: The context the answer was produced from.
    """

    __slots__ = ("answer", "context", "message", "query")

    def __init__(
        self,
        *,
        message: cl.Message,
        query: str,
        answer: Answer,
        context: PromptContext,
    ) -> None:
        """Record one answered question.

        Args:
            message: The answer message.
            query: The question that was asked.
            answer: The answer that was produced.
            context: The context the answer was produced from.
        """
        self.message = message
        self.query = query
        self.answer = answer
        self.context = context


def build_feedback_log(settings: Settings) -> JsonlFeedbackLog:
    """Open the log that ratings are appended to.

    Args:
        settings: Application settings.

    Returns:
        The feedback log.
    """
    return JsonlFeedbackLog(settings.feedback_path)


@cl.on_chat_start
async def start() -> None:
    """Load the pipeline once per session and report what is configured.

    The embedding model and cross-encoder are expensive to load and hold no
    per-request state, so they are built here and reused for every message.
    """
    try:
        settings = load_settings()
        orchestrator = await asyncio.to_thread(build_retrieval_orchestrator, settings)
        chat_model = build_chat_model(settings)
    except (ConfigurationError, ModelUnavailableError) as exc:
        await cl.Message(content=f"**This interface cannot start.**\n\n{exc}").send()
        return

    cl.user_session.set(_SETTINGS, settings)
    cl.user_session.set(_ORCHESTRATOR, orchestrator)
    cl.user_session.set(_CHAT_MODEL, chat_model)
    cl.user_session.set(_FEEDBACK_LOG, build_feedback_log(settings))
    cl.user_session.set(_TURNS, {})

    reranker = "cross-encoder" if settings.reranker_model_path else "vector only"
    await cl.Message(
        content=(
            f"Ready. Answering with **{chat_model.model_name}** over "
            f"**{settings.top_k}** passages ({reranker}), using prompt "
            f"`{settings.prompt_name}/{settings.prompt_version}`.\n\n"
            "Answers are drawn only from the indexed documents. Rate them with "
            "the buttons underneath — the ratings are what will eventually let "
            "retrieval quality be measured rather than assumed."
        )
    ).send()


@cl.on_message
async def answer(message: cl.Message) -> None:
    """Retrieve, answer, cite, and offer the rating buttons.

    Args:
        message: The question the reader asked.
    """
    orchestrator: RetrievalOrchestrator | None = cl.user_session.get(_ORCHESTRATOR)
    settings: Settings | None = cl.user_session.get(_SETTINGS)
    chat_model: BaseChatModel | None = cl.user_session.get(_CHAT_MODEL)
    if orchestrator is None or settings is None or chat_model is None:
        await cl.Message(content="The interface did not start; see above.").send()
        return

    query = message.content.strip()
    try:
        async with cl.Step(name="Retrieving", type="retrieval") as step:
            step.input = query
            context = await asyncio.to_thread(
                orchestrator.retrieve, query, top_k=settings.top_k
            )
            step.output = f"{len(context.chunks)} passages"
    except ValueError as exc:
        await cl.Message(content=f"That question was rejected: {exc}").send()
        return
    except Exception:
        logger.exception("Retrieval failed.")
        await cl.Message(
            content="Retrieval failed. Check that the database is reachable."
        ).send()
        return

    if not context.chunks:
        await cl.Message(
            content=(
                "Nothing in the indexed documents matches that question, so the "
                "model was not asked. An answer with no sources would be a guess."
            )
        ).send()
        return

    streamed = await _stream_answer(chat_model, context)
    if streamed is None:
        return

    reply, answer = streamed
    await _offer_rating(reply, query=query, answer=answer, context=context)


async def _stream_answer(
    chat_model: BaseChatModel, context: PromptContext
) -> tuple[cl.Message, Answer] | None:
    """Stream one answer into the UI, reasoning apart from the answer.

    The model client is a synchronous generator, so each fragment is pulled in
    a worker thread. That keeps the event loop free without pushing ``async``
    into the package.

    Two things here exist to make the transcript read in the order the work
    happened — reasoning, then answer. The step is entered as a context
    manager, because that is the only path on which Chainlit stamps a step's
    start time, and a step without one is ordered by its last update, which
    lands it *below* the answer it preceded. And the answer message is not
    built until the first answer fragment arrives, so it cannot be timestamped
    earlier than the reasoning it followed.

    Both are lazy, so a model that does not reason, or one asked not to,
    leaves no empty step behind.

    Args:
        chat_model: The model to answer with.
        context: The assembled retrieval context.

    Returns:
        The message the answer was streamed into and the completed answer, or
        ``None`` if generation failed and the failure has already been
        reported to the reader.
    """
    deltas = chat_model.stream(context)
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    reply: cl.Message | None = None
    reasoning_step: cl.Step | None = None
    reasoning = AsyncExitStack()

    try:
        while (delta := await _next(deltas)) is not None:
            if delta.reasoning:
                if reasoning_step is None:
                    reasoning_step = cl.Step(
                        name="Reasoning", type="llm", default_open=False
                    )
                    await reasoning.enter_async_context(reasoning_step)
                reasoning_parts.append(delta.text)
                await reasoning_step.stream_token(delta.text)
                continue

            if reply is None:
                await reasoning.aclose()
                reply = cl.Message(content="", elements=_source_elements(context))
            answer_parts.append(delta.text)
            await reply.stream_token(delta.text)
    except GenerationError as exc:
        await cl.Message(content=f"**No answer.** {exc}").send()
        return None
    finally:
        await reasoning.aclose()

    if reply is None:
        reply = cl.Message(
            content=(
                "The model reasoned but produced no answer. Its reasoning is "
                "above; try the question again, or a shorter one."
            ),
            elements=_source_elements(context),
        )

    return reply, Answer(
        text="".join(answer_parts),
        reasoning="".join(reasoning_parts),
        model_name=chat_model.model_name,
        prompt_metadata=context.prompt_metadata,
    )


async def _next(deltas: Iterator[AnswerDelta]) -> AnswerDelta | None:
    """Pull the next fragment without blocking the event loop.

    Args:
        deltas: The stream being consumed.

    Returns:
        The next fragment, or ``None`` when the stream is exhausted.
    """
    return await asyncio.to_thread(next, deltas, None)


def _source_elements(context: PromptContext) -> list[cl.Text]:
    """Render each cited passage as an inspectable side element.

    Numbering is the augmenter's, one-based, so a ``[2]`` in the answer is the
    element labelled ``[2]`` here.

    Passage text is placed in a fenced block. It is untrusted document
    content, and the interface renders Markdown: a document containing image
    or link syntax should be shown, not obeyed.

    Args:
        context: The context the answer was produced from.

    Returns:
        One text element per passage, in prompt order.
    """
    elements: list[cl.Text] = []
    for position, chunk in enumerate(context.chunks, start=1):
        source = chunk.chunk
        elements.append(
            cl.Text(
                name=f"[{position}]",
                display="side",
                content=(
                    f"page {source.page_number} · "
                    f"{source.section or 'no section'} · "
                    f"score {chunk.rerank_score:.3f}\n\n"
                    f"```text\n{source.text}\n```\n\n"
                    f"document `{source.document_id}`"
                ),
            )
        )
    return elements


async def _offer_rating(
    reply: cl.Message, *, query: str, answer: Answer, context: PromptContext
) -> None:
    """Close the streamed answer and attach the rating buttons.

    One ``send`` ends the stream and delivers the citations and buttons
    together, so the answer is never briefly readable without its sources.

    Args:
        reply: The answer message being rated.
        query: The question that was asked.
        answer: The answer that was produced.
        context: The context it was produced from.
    """
    reply.actions = [
        cl.Action(
            name="vote_up",
            payload={"turn": reply.id},
            label="Helpful",
            icon="thumbs-up",
            tooltip="This answer used the sources well.",
        ),
        cl.Action(
            name="vote_down",
            payload={"turn": reply.id},
            label="Not helpful",
            icon="thumbs-down",
            tooltip="Wrong, unsupported, or the sources were not relevant.",
        ),
    ]
    await reply.send()

    turns: dict[str, Any] = cl.user_session.get(_TURNS) or {}
    turns[reply.id] = _Turn(message=reply, query=query, answer=answer, context=context)
    cl.user_session.set(_TURNS, turns)


async def _record(action: cl.Action, vote: Vote) -> None:
    """Write one rating and retire the buttons.

    Args:
        action: The button that was clicked.
        vote: The rating it represents.
    """
    turns: dict[str, Any] = cl.user_session.get(_TURNS) or {}
    turn: _Turn | None = turns.get(str(action.payload.get("turn")))
    log: JsonlFeedbackLog | None = cl.user_session.get(_FEEDBACK_LOG)
    if turn is None or log is None:
        await cl.Message(content="That answer is no longer available to rate.").send()
        return

    record = FeedbackRecord.build(vote, turn.query, turn.answer, turn.context)
    try:
        await asyncio.to_thread(log.record, record)
    except OSError:
        logger.exception("Could not write feedback to %s.", log.path)
        await cl.Message(
            content="The rating could not be saved; the feedback log is unwritable."
        ).send()
        return

    del turns[str(action.payload.get("turn"))]
    cl.user_session.set(_TURNS, turns)
    await turn.message.remove_actions()
    await cl.Message(content="Recorded — thank you.").send()


@cl.action_callback("vote_up")
async def vote_up(action: cl.Action) -> None:
    """Record a positive rating.

    Args:
        action: The button that was clicked.
    """
    await _record(action, "up")


@cl.action_callback("vote_down")
async def vote_down(action: cl.Action) -> None:
    """Record a negative rating.

    Args:
        action: The button that was clicked.
    """
    await _record(action, "down")
