"""A chat interface over an OpenAI-language RAG provider.

This is a launcher, not a layer. It talks to the provider over HTTP in the
OpenAI wire dialect and has no dependency on the ``rag`` package: the whole
``app-openai/`` directory can be copied into another repository, given
``RAG_OPENAI_BASE_URL`` and ``RAG_OPENAI_API_KEY``, and run against a deployed
provider.

It is the sibling of ``app/`` — same Chainlit UI, same streaming, reasoning
step, citation panels and rating buttons — differing in the wire language it
speaks and, deliberately, in being a little leaner internally. The one UI
addition over ``app/`` is a file-upload affordance, so ingestion can be shown.

Run it with::

    chainlit run src/rag_chat_openai/main.py

With no provider reachable — or with ``RAG_OPENAI_DEMO`` set — it starts in demo
mode: a canned OpenAI-shaped chunk stream is decoded through the real client, so
the interface (including the "OpenAI wire trace" step) can be shown without a
backend. Ratings are not recorded in that mode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import chainlit as cl
import httpx

from rag_chat_openai.client import (
    ApiError,
    OpenAIRagClient,
    _StreamDecoder,
    decode_lines,
)
from rag_chat_openai.config import AppConfig
from rag_chat_openai.demo import demo_lines
from rag_chat_openai.models import Citation, Event

logger = logging.getLogger(__name__)

#: Session keys. Named once so a typo cannot silently read back ``None``.
_CONFIG = "config"
_CLIENT = "client"
_DEMO = "demo"
_TURNS = "turns"


class _Turn:
    """One answered question, held until it is rated or the session ends.

    A rating arrives long after the answer was streamed, as a click carrying
    nothing but a message id. Everything feedback needs is kept here so the
    rating never has to be reconstructed from what is on screen.

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
        citations: Provenance of the passages the answer was given.
    """

    __slots__ = ("answer", "citations", "message", "model_name", "query")

    def __init__(
        self,
        *,
        message: cl.Message,
        query: str,
        answer: str,
        model_name: str,
        citations: tuple[Citation, ...],
    ) -> None:
        """Record one answered question.

        Args:
            message: The answer message.
            query: The question that was asked.
            answer: The answer text that was produced.
            model_name: The model that produced it.
            citations: Provenance of the passages the answer was given.
        """
        self.message = message
        self.query = query
        self.answer = answer
        self.model_name = model_name
        self.citations = citations


@cl.on_chat_start
async def start() -> None:
    """Open a client, check the provider, and report what is configured.

    When the provider cannot be reached, or ``RAG_OPENAI_DEMO`` is set, the
    session goes into demo mode instead of refusing to start.
    """
    config = AppConfig.from_env()
    client = OpenAIRagClient(
        config.base_url,
        config.api_key,
        model=config.model,
        feedback_url=config.feedback_url,
    )
    cl.user_session.set(_CONFIG, config)
    cl.user_session.set(_CLIENT, client)
    cl.user_session.set(_TURNS, {})

    reason: str | None = None
    model_name: str | None = None
    if config.demo:
        reason = "`RAG_OPENAI_DEMO` is set"
    else:
        try:
            model_name = await client.models()
        except httpx.HTTPStatusError as exc:
            reason = (
                f"the provider at `{config.base_url}` returned HTTP "
                f"{exc.response.status_code}"
            )
        except httpx.HTTPError:
            reason = f"the provider at `{config.base_url}` is unreachable"

    if reason is not None or model_name is None:
        cl.user_session.set(_DEMO, True)
        await cl.Message(
            content=(
                f"**Demo mode** — {reason}. Answers below are canned and "
                "ratings are not recorded. The stream is still decoded through "
                "the real client, and the raw chunks are shown in the "
                '"OpenAI wire trace" step. Set `RAG_OPENAI_BASE_URL` and reload '
                "to ask for real."
            )
        ).send()
        return

    cl.user_session.set(_DEMO, False)
    await cl.Message(content=_ready_banner(model_name)).send()


def _ready_banner(model_name: str) -> str:
    """Compose the startup banner.

    Args:
        model_name: The model the provider will answer with.

    Returns:
        The banner text.
    """
    return (
        f"Ready. Answering with **{model_name}**.\n\n"
        "Answers are drawn only from the provider's indexed documents. Rate "
        "them with the buttons underneath — the ratings are what will "
        "eventually let retrieval quality be measured rather than assumed."
    )


@cl.on_chat_end
async def stop() -> None:
    """Close the HTTP client when the session ends."""
    client: OpenAIRagClient | None = cl.user_session.get(_CLIENT)
    if client is not None:
        await client.aclose()


@cl.on_message
async def answer(message: cl.Message) -> None:
    """Stream an answer, cite it, and offer the rating buttons.

    A message carrying file attachments is treated as an ingestion request
    instead.

    Args:
        message: The question the reader asked, or a document to ingest.
    """
    client: OpenAIRagClient | None = cl.user_session.get(_CLIENT)
    if client is None:
        await cl.Message(content="The interface did not start; see above.").send()
        return

    uploads = [
        element
        for element in (message.elements or [])
        if getattr(element, "path", None)
    ]
    if uploads:
        await _handle_uploads(client, uploads)
        return

    query = message.content.strip()
    demo = bool(cl.user_session.get(_DEMO))
    decoder = _StreamDecoder()
    if demo:
        source: AsyncIterator[Event] = decode_lines(demo_lines(query), decoder)
    else:
        source = client.answer_stream(query, decoder)

    state = await _stream_answer(source)
    if demo:
        await _render_trace(decoder)
    if state is None:
        return

    await _offer_rating(state, query=query)


async def _handle_uploads(client: OpenAIRagClient, uploads: list[cl.Element]) -> None:
    """Ingest each attached document, or fake it in demo mode.

    Args:
        client: The provider client.
        uploads: The message elements that carry a local file path.
    """
    demo = bool(cl.user_session.get(_DEMO))
    for element in uploads:
        name = element.name or "document"
        if demo:
            await cl.Message(
                content=(
                    f"**Indexed** `{name}` → `demo-0000` · 12 chunks "
                    "(demo — not stored)."
                )
            ).send()
            continue
        try:
            outcome = await client.ingest(Path(str(element.path)))
        except httpx.HTTPError:
            logger.exception("Ingestion failed for %s.", name)
            await cl.Message(
                content=f"`{name}` could not be ingested; the provider rejected it."
            ).send()
            continue
        file_id = outcome.get("id", "unknown")
        await cl.Message(content=f"**Indexed** `{name}` → `{file_id}`.").send()


async def _render_trace(decoder: _StreamDecoder) -> None:
    """Show the raw OpenAI chunks the decoder saw, in a collapsed step.

    Args:
        decoder: The decoder that drove the just-finished stream.
    """
    if not decoder.trace:
        return
    lines: list[str] = []
    for raw in decoder.trace:
        if raw == "[DONE]":
            lines.append("data: [DONE]")
            continue
        try:
            lines.append("data: " + json.dumps(json.loads(raw), indent=2))
        except json.JSONDecodeError:
            lines.append("data: " + raw)
    async with cl.Step(name="OpenAI wire trace", type="tool") as step:
        step.output = "\n\n".join(lines)


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

    Hand-written rather than a dataclass for the same reason as :class:`_Turn`.
    """

    __slots__ = (
        "answer_parts",
        "citations",
        "done",
        "elements",
        "model_name",
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
        self.done = False
        self.model_name = "unknown"

    @property
    def answer_text(self) -> str:
        """The answer text streamed so far, joined."""
        return "".join(self.answer_parts)

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


async def _stream_answer(source: AsyncIterator[Event]) -> _StreamState | None:
    """Stream one answer into the UI, reasoning apart from the answer.

    Args:
        source: The decoded event stream, from the client or the demo decoder.

    Returns:
        The accumulated stream state when there is an answer to rate, or
        ``None`` when the stream ended without one — the reason has already been
        shown to the reader.
    """
    state = _StreamState()
    try:
        async for event in source:
            if event.kind == "citations":
                state.citations = event.citations
                state.elements = _source_elements(event.citations)
            elif event.kind == "reasoning":
                await state.add_reasoning(event.text)
            elif event.kind == "answer":
                await state.add_answer(event.text)
            elif event.kind == "done":
                state.model_name = event.text
                state.done = True
            else:  # "error"
                await state.reasoning.aclose()
                await cl.Message(content=f"**No answer.** {event.text}").send()
                return None
    except ApiError as exc:
        await cl.Message(content=f"That question was rejected: {exc}").send()
        return None
    except httpx.HTTPError:
        logger.exception("The answer stream failed.")
        await cl.Message(
            content="The connection to the provider failed. Check that it is running."
        ).send()
        return None
    finally:
        await state.reasoning.aclose()

    if not state.done:
        await cl.Message(
            content="The provider ended the stream without finishing the answer."
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
    return state


def _source_elements(citations: tuple[Citation, ...]) -> list[cl.Text]:
    """Render each cited passage as an inspectable side element.

    Numbering is the provider's, one-based, so a ``[2]`` in the answer is the
    element labelled ``[2]`` here. The passage text itself stays with the
    provider — the app is only given provenance — so these panels show where a
    passage came from, not what it said.

    Args:
        citations: The provenance of each passage, in prompt order.

    Returns:
        One text element per passage.
    """
    elements: list[cl.Text] = []
    for citation in citations:
        page = f"page {citation.page_number}" if citation.page_number else "page —"
        score = f"score {citation.score:.3f}" if citation.score else "score —"
        elements.append(
            cl.Text(
                name=f"[{citation.position}]",
                display="side",
                content=(
                    f"{page} · {citation.section or 'no section'} · {score}\n\n"
                    f"document `{citation.document_id}`"
                ),
            )
        )
    return elements


async def _offer_rating(state: _StreamState, *, query: str) -> None:
    """Close the streamed answer and attach the rating buttons.

    One ``send`` ends the stream and delivers the citations and buttons
    together, so the answer is never briefly readable without its sources.

    Args:
        state: The finished stream state. ``_stream_answer`` guarantees
            ``state.reply`` is set on a non-``None`` return.
        query: The question that was asked.
    """
    reply = state.reply
    if reply is None:  # unreachable after _stream_answer; keeps the type-checker honest
        return
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
        answer=state.answer_text,
        model_name=state.model_name,
        citations=state.citations,
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
        client: OpenAIRagClient | None = cl.user_session.get(_CLIENT)
        if client is None:
            await cl.Message(content="The interface did not start; see above.").send()
            return
        try:
            outcome = await client.feedback(
                vote=vote,
                query=turn.query,
                answer=turn.answer,
                model_name=turn.model_name,
                citations=turn.citations,
            )
        except httpx.HTTPError:
            logger.exception("Feedback POST failed.")
            await cl.Message(
                content="The rating could not be saved; the provider rejected it."
            ).send()
            return
        message = (
            "Recorded — thank you."
            if outcome == "recorded"
            else "Noted — no feedback endpoint is configured, so nothing was stored."
        )

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
