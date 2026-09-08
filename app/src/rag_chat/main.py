"""A chat interface over the rag-study API.

This is a launcher, not a layer. It talks to the API service over HTTP and has
no dependency on the ``rag`` package: the whole ``app/`` directory can be
copied into another repository, given ``RAG_API_URL`` and ``RAG_API_KEY``, and
run against a deployed service.

It bridges two worlds. Chainlit is asynchronous; the API stream is consumed
with ``async for``. The blocking work all lives on the service side now, so
there is nothing here to push into a worker thread.

Run it with::

    chainlit run src/rag_chat/main.py

With no service reachable — or with ``RAG_CHAT_DEMO`` set — it starts in demo
mode: a canned answer streams through the real UI so the interface can be
shown without a backend. Ratings are not recorded in that mode.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

import chainlit as cl
import httpx

from rag_chat.client import ApiError, RagApiClient
from rag_chat.config import AppConfig
from rag_chat.demo import demo_events
from rag_chat.models import (
    Citation,
    CitationsEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    NoContextEvent,
    PromptId,
    StreamEvent,
)

logger = logging.getLogger(__name__)

#: Session keys. Named once so a typo cannot silently read back ``None``.
_CONFIG = "config"
_CLIENT = "client"
_DEMO = "demo"
_TURNS = "turns"


class _Turn:
    """One answered question, held until it is rated or the session ends.

    A rating arrives long after the answer was streamed, as a click carrying
    nothing but a message id. Everything ``POST /feedback`` needs is kept here
    so the rating never has to be reconstructed from what is on screen.

    This is written out by hand rather than as a dataclass, and must stay that
    way. Chainlit executes this file without registering it in
    ``sys.modules``, and ``dataclasses`` resolves the string annotations that
    ``from __future__ import annotations`` produces by looking the defining
    module up there. It is not found, and the decorator raises at import time.

    Attributes:
        message: The answer message, so its buttons can be retired.
        query: The question that was asked.
        answer: The answer text that was produced.
        model_name: The model that produced it.
        prompt: The prompt template that built its context.
        citations: Provenance of the passages the answer was given.
    """

    __slots__ = ("answer", "citations", "message", "model_name", "prompt", "query")

    def __init__(
        self,
        *,
        message: cl.Message,
        query: str,
        answer: str,
        model_name: str,
        prompt: PromptId,
        citations: tuple[Citation, ...],
    ) -> None:
        """Record one answered question.

        Args:
            message: The answer message.
            query: The question that was asked.
            answer: The answer text that was produced.
            model_name: The model that produced it.
            prompt: The prompt template that built its context.
            citations: Provenance of the passages the answer was given.
        """
        self.message = message
        self.query = query
        self.answer = answer
        self.model_name = model_name
        self.prompt = prompt
        self.citations = citations


@cl.on_chat_start
async def start() -> None:
    """Open a client, check the service, and report what is configured.

    When the service cannot be reached, or ``RAG_CHAT_DEMO`` is set, the
    session goes into demo mode instead of refusing to start.
    """
    config = AppConfig.from_env()
    client = RagApiClient(config.api_url, config.api_key)
    cl.user_session.set(_CONFIG, config)
    cl.user_session.set(_CLIENT, client)
    cl.user_session.set(_TURNS, {})

    reason: str | None = None
    health = None
    if config.demo:
        reason = "`RAG_CHAT_DEMO` is set"
    else:
        try:
            health = await client.health()
        except httpx.HTTPStatusError as exc:
            reason = (
                f"the service at `{config.api_url}` returned HTTP "
                f"{exc.response.status_code}"
            )
        except httpx.HTTPError:
            reason = f"the service at `{config.api_url}` is unreachable"

    if reason is not None or health is None:
        cl.user_session.set(_DEMO, True)
        await cl.Message(
            content=(
                f"**Demo mode** — {reason}. Answers below are canned and "
                "ratings are not recorded. Start the API service (`just serve` "
                "in the rag-study repo) and reload to ask for real."
            )
        ).send()
        return

    cl.user_session.set(_DEMO, False)
    await cl.Message(
        content=(
            f"Ready. Answering with **{health.model}** over **{health.top_k}** "
            f"passages ({health.reranker}), using prompt `{health.prompt}`.\n\n"
            "Answers are drawn only from the indexed documents. Rate them with "
            "the buttons underneath — the ratings are what will eventually let "
            "retrieval quality be measured rather than assumed."
        )
    ).send()


@cl.on_chat_end
async def stop() -> None:
    """Close the HTTP client when the session ends."""
    client: RagApiClient | None = cl.user_session.get(_CLIENT)
    if client is not None:
        await client.aclose()


@cl.on_message
async def answer(message: cl.Message) -> None:
    """Stream an answer, cite it, and offer the rating buttons.

    Args:
        message: The question the reader asked.
    """
    client: RagApiClient | None = cl.user_session.get(_CLIENT)
    config: AppConfig | None = cl.user_session.get(_CONFIG)
    if client is None or config is None:
        await cl.Message(content="The interface did not start; see above.").send()
        return

    query = message.content.strip()
    if cl.user_session.get(_DEMO):
        source: AsyncIterator[StreamEvent] = demo_events(query)
    else:
        source = client.answer_stream(query, top_k=config.top_k)

    streamed = await _stream_answer(source)
    if streamed is None:
        return

    reply, answer_text, done, citations = streamed
    await _offer_rating(
        reply, query=query, answer=answer_text, done=done, citations=citations
    )


class _StreamState:
    """Mutable state accumulated while a single answer streams in.

    Two things here make the transcript read in the order the work happened —
    reasoning, then answer. The reasoning step is entered as a context
    manager, because that is the only path on which Chainlit stamps a step's
    start time; a step without one is ordered by its last update, which lands
    it *below* the answer it preceded. And the answer message is not built
    until the first answer fragment arrives, so it cannot be timestamped
    earlier than the reasoning it followed. Both are lazy, so a model that
    does not reason leaves no empty step behind.
    """

    __slots__ = (
        "answer_parts",
        "citations",
        "done",
        "elements",
        "reasoning",
        "reasoning_step",
        "reply",
    )

    def __init__(self) -> None:
        """Start with nothing streamed yet."""
        self.citations: tuple[Citation, ...] = ()
        self.elements: list[cl.Text] = []
        self.answer_parts: list[str] = []
        self.reply: cl.Message | None = None
        self.reasoning_step: cl.Step | None = None
        self.reasoning = AsyncExitStack()
        self.done: DoneEvent | None = None

    async def add_reasoning(self, text: str) -> None:
        """Stream a reasoning fragment into a lazily created step.

        Args:
            text: The fragment.
        """
        if self.reasoning_step is None:
            self.reasoning_step = cl.Step(
                name="Reasoning", type="llm", default_open=False
            )
            await self.reasoning.enter_async_context(self.reasoning_step)
        await self.reasoning_step.stream_token(text)

    async def add_answer(self, text: str) -> None:
        """Stream an answer fragment into a lazily created message.

        Args:
            text: The fragment.
        """
        if self.reply is None:
            await self.reasoning.aclose()
            self.reply = cl.Message(content="", elements=self.elements)
        self.answer_parts.append(text)
        await self.reply.stream_token(text)


async def _terminal_message(event: NoContextEvent | ErrorEvent) -> None:
    """Show the message for a stream that ended without an answer.

    Args:
        event: The ``no_context`` or ``error`` event that ended the stream.
    """
    if isinstance(event, NoContextEvent):
        await cl.Message(
            content=(
                "Nothing in the indexed documents matches that question, so "
                "the model was not asked. An answer with no sources would be "
                "a guess."
            )
        ).send()
    else:
        await cl.Message(content=f"**No answer.** {event.message}").send()


async def _consume(source: AsyncIterator[StreamEvent], state: _StreamState) -> bool:
    """Drive the stream into ``state`` until it ends.

    Args:
        source: The stream of events from the API or the demo generator.
        state: The state to accumulate into.

    Returns:
        ``True`` if the stream produced an answer, ``False`` if it ended on a
        ``no_context`` or ``error`` event (whose message has been shown).
    """
    async for event in source:
        match event:
            case CitationsEvent():
                state.citations = event.items
                state.elements = _source_elements(event.items)
            case DeltaEvent(reasoning=True):
                await state.add_reasoning(event.text)
            case DeltaEvent():
                await state.add_answer(event.text)
            case DoneEvent():
                state.done = event
            case _:
                await state.reasoning.aclose()
                await _terminal_message(event)
                return False
    return True


async def _stream_answer(
    source: AsyncIterator[StreamEvent],
) -> tuple[cl.Message, str, DoneEvent, tuple[Citation, ...]] | None:
    """Stream one answer into the UI, reasoning apart from the answer.

    Args:
        source: The stream of events from the API or the demo generator.

    Returns:
        The message the answer was streamed into, the answer text, the
        ``done`` event, and the citations — or ``None`` when there was no
        answer to rate and the reason has already been shown to the reader.
    """
    state = _StreamState()
    try:
        produced = await _consume(source, state)
    except ApiError as exc:
        await cl.Message(content=f"That question was rejected: {exc}").send()
        return None
    except httpx.HTTPError:
        logger.exception("The answer stream failed.")
        await cl.Message(
            content=(
                "The connection to the API service failed. Check that it is running."
            )
        ).send()
        return None
    finally:
        await state.reasoning.aclose()

    if not produced:
        return None
    if state.done is None:
        await cl.Message(
            content="The service ended the stream without finishing the answer."
        ).send()
        return None
    if state.reply is None:
        state.reply = cl.Message(
            content=(
                "The model reasoned but produced no answer. Its reasoning is "
                "above; try the question again, or a shorter one."
            ),
            elements=state.elements,
        )
    return state.reply, "".join(state.answer_parts), state.done, state.citations


def _source_elements(citations: tuple[Citation, ...]) -> list[cl.Text]:
    """Render each cited passage as an inspectable side element.

    Numbering is the service's, one-based, so a ``[2]`` in the answer is the
    element labelled ``[2]`` here. The passage text itself stays on the
    service — it is untrusted document content and the app is only given its
    provenance — so these panels show where a passage came from, not what it
    said.

    Args:
        citations: The provenance of each passage, in prompt order.

    Returns:
        One text element per passage.
    """
    elements: list[cl.Text] = []
    for citation in citations:
        elements.append(
            cl.Text(
                name=f"[{citation.position}]",
                display="side",
                content=(
                    f"page {citation.page_number} · "
                    f"{citation.section or 'no section'} · "
                    f"score {citation.score:.3f}\n\n"
                    f"document `{citation.document_id}`"
                ),
            )
        )
    return elements


async def _offer_rating(
    reply: cl.Message,
    *,
    query: str,
    answer: str,
    done: DoneEvent,
    citations: tuple[Citation, ...],
) -> None:
    """Close the streamed answer and attach the rating buttons.

    One ``send`` ends the stream and delivers the citations and buttons
    together, so the answer is never briefly readable without its sources.

    Args:
        reply: The answer message being rated.
        query: The question that was asked.
        answer: The answer text that was produced.
        done: The stream's ``done`` event, carrying the model and prompt id.
        citations: Provenance of the passages the answer was given.
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
    turns[reply.id] = _Turn(
        message=reply,
        query=query,
        answer=answer,
        model_name=done.model_name,
        prompt=done.prompt,
        citations=citations,
    )
    cl.user_session.set(_TURNS, turns)


async def _record(action: cl.Action, vote: str) -> None:
    """Send one rating and retire the buttons.

    Args:
        action: The button that was clicked.
        vote: The rating it represents, ``"up"`` or ``"down"``.
    """
    turns: dict[str, Any] = cl.user_session.get(_TURNS) or {}
    key = str(action.payload.get("turn"))
    turn: _Turn | None = turns.get(key)
    if turn is None:
        await cl.Message(content="That answer is no longer available to rate.").send()
        return

    if cl.user_session.get(_DEMO):
        message = "Recorded (demo — not saved)."
    else:
        client: RagApiClient | None = cl.user_session.get(_CLIENT)
        if client is None:
            await cl.Message(content="The interface did not start; see above.").send()
            return
        try:
            await client.feedback(
                vote=vote,
                query=turn.query,
                answer=turn.answer,
                model_name=turn.model_name,
                prompt=turn.prompt,
                citations=turn.citations,
            )
        except httpx.HTTPError:
            logger.exception("Feedback POST failed.")
            await cl.Message(
                content="The rating could not be saved; the service rejected it."
            ).send()
            return
        message = "Recorded — thank you."

    del turns[key]
    cl.user_session.set(_TURNS, turns)
    await turn.message.remove_actions()
    await cl.Message(content=message).send()


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
