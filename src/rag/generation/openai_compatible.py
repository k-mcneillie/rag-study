"""An answering model behind an OpenAI-compatible chat completions API.

The second implementation of the same interface, and the point of it. It is
what makes the claim that generation is swappable checkable rather than
asserted: a completely different wire protocol, a completely different
streaming format, no change anywhere else in the system.

It also covers most of what a reader is likely to reach for next. vLLM, LM
Studio, llama.cpp's server, OpenAI, Together, Groq, and OpenRouter all speak
this protocol, as does Ollama itself on ``/v1``. Choosing between them is a
base URL, a model name, and a credential.

**vLLM** is supported through this client and is a required target, but say
plainly what that means: it is supported by protocol, and has not been
exercised against a running vLLM server. vLLM implements the same endpoint,
the same server-sent event framing, and — when started with a reasoning parser
such as ``--reasoning-parser deepseek_r1`` — the same ``reasoning_content``
field this client already reads. Its base URL conventionally ends in ``/v1``,
which is handled. If it does misbehave, the fault will be in this file alone.

Reasoning is the one thing this protocol handles worse than Ollama's own. Some
services return it in a ``reasoning`` or ``reasoning_content`` field, which is
read here when present; others inline it in ``<think>`` tags, which are split
out as they stream. For a reasoning model on Ollama, the native client is
preferable for exactly that reason.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from rag.config import GenerationSettings
from rag.domain.models import AnswerDelta, PromptContext
from rag.generation.interfaces import BaseChatModel, GenerationError
from rag.generation.transport import stream_lines

logger = logging.getLogger(__name__)

#: The protocol's streaming chat endpoint, relative to its base URL, and the
#: same endpoint for a base URL that already includes the version segment.
#: Both spellings are common in the wild — OpenAI and vLLM are usually
#: configured with the ``/v1`` included, Ollama's compatible endpoint without
#: it — and posting to ``/v1/v1/chat/completions`` fails as a bare 404 that
#: reads like a missing model.
_CHAT_PATH = "/v1/chat/completions"
_VERSIONED_CHAT_PATH = "/chat/completions"
_VERSION_SEGMENT = "/v1"

#: Server-sent event framing: the prefix on each data line, and the sentinel
#: that ends the stream.
_DATA_PREFIX = "data:"
_DONE = "[DONE]"

#: Fields that carry reasoning when a service separates it. Checked in order.
_REASONING_FIELDS = ("reasoning_content", "reasoning")

#: Tags used by services that inline reasoning in the answer instead.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_length(text: str, tag: str) -> int:
    """Measure how much of ``text``'s tail could be the start of ``tag``.

    Args:
        text: The buffered text.
        tag: The tag being watched for.

    Returns:
        The length of the longest suffix of ``text`` that is a proper prefix
        of ``tag``, or zero when no suffix could become one.
    """
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


class OpenAICompatibleChatModel(BaseChatModel):
    """Answers by streaming from an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        settings: GenerationSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Prepare a client for an OpenAI-compatible service.

        Args:
            settings: Where the service is, how to authenticate, and how it
                should sample.
            transport: An HTTP transport to use instead of the default. Only
                tests supply this.
        """
        self._settings = settings
        self._transport = transport
        self._in_reasoning = False
        self._pending = ""

    @property
    def model_name(self) -> str:
        """Identify the configured model.

        Returns:
            The model name as configured.
        """
        return self._settings.model

    def _request_body(self, context: PromptContext) -> dict[str, Any]:
        """Build the request payload for one question.

        The rendered prompt is sent as a single user message, exactly as the
        augmenter produced it, for the same reason as in every other client:
        the template's instructions and its source-material markers are one
        structure that this component does not own.

        ``num_ctx`` has no equivalent in this protocol and is not sent. A
        service that sizes its own context window is trusted to do so.

        Args:
            context: Assembled retrieval context.

        Returns:
            The JSON body to post.
        """
        return {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": context.rendered_text}],
            "stream": True,
            "temperature": self._settings.temperature,
        }

    def _deltas(self, line: str) -> Iterator[AnswerDelta]:
        """Interpret one server-sent event from the stream.

        Args:
            line: A single line of the response body.

        Yields:
            The fragments the event carries.

        Raises:
            GenerationError: If the event is not JSON, or reports an error.
        """
        if not line.startswith(_DATA_PREFIX):
            return
        data = line[len(_DATA_PREFIX) :].strip()
        if data == _DONE:
            return

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"{self._settings.base_url} returned a malformed response "
                f"for model {self._settings.model}."
            ) from exc

        if error := payload.get("error"):
            detail = error.get("message") if isinstance(error, dict) else error
            raise GenerationError(
                f"Model {self._settings.model} failed at "
                f"{self._settings.base_url}: {detail}"
            )

        choices = payload.get("choices") or []
        if not choices:
            return
        delta = choices[0].get("delta") or {}

        for name in _REASONING_FIELDS:
            if reasoning := delta.get(name):
                yield AnswerDelta(text=reasoning, reasoning=True)
        if content := delta.get("content"):
            yield from self._split_inline_reasoning(content)

    def _split_inline_reasoning(self, content: str) -> Iterator[AnswerDelta]:
        """Separate inlined ``<think>`` reasoning from answer text.

        Services that do not model reasoning as its own field emit it inside
        tags in the content stream. A tag can arrive split across fragments —
        ``"<thi"`` then ``"nk>"`` — so text that might still turn out to be
        the start of one is held back rather than emitted. Without that, the
        model's private reasoning is shown to the reader as though it were the
        answer, which is the failure this whole split exists to prevent.

        Args:
            content: One fragment of content text.

        Yields:
            The fragment, split at any tag boundary, minus any trailing text
            that could still become a tag.
        """
        self._pending += content

        while True:
            tag = _THINK_CLOSE if self._in_reasoning else _THINK_OPEN
            position = self._pending.find(tag)
            if position >= 0:
                if head := self._pending[:position]:
                    yield AnswerDelta(text=head, reasoning=self._in_reasoning)
                self._pending = self._pending[position + len(tag) :]
                self._in_reasoning = not self._in_reasoning
                continue

            held = _partial_tag_length(self._pending, tag)
            emit = self._pending[: len(self._pending) - held]
            self._pending = self._pending[len(self._pending) - held :]
            if emit:
                yield AnswerDelta(text=emit, reasoning=self._in_reasoning)
            return

    def _flush(self) -> Iterator[AnswerDelta]:
        """Emit text held back in case it was the start of a tag.

        Yields:
            Whatever is still buffered when the stream ends. Text withheld as
            a possible tag was, in the end, just text.
        """
        if self._pending:
            yield AnswerDelta(text=self._pending, reasoning=self._in_reasoning)
            self._pending = ""

    def stream(self, context: PromptContext) -> Iterator[AnswerDelta]:
        """Answer the question in ``context``, a fragment at a time.

        Args:
            context: Assembled retrieval context, sent unchanged.

        Yields:
            Fragments of reasoning and answer, in the order they arrive.

        Raises:
            GenerationError: If the service is unreachable, rejects the
                request, or streams something unintelligible.
        """
        logger.info(
            "Generating with model %s at %s.",
            self._settings.model,
            self._settings.base_url,
        )
        self._in_reasoning = False
        self._pending = ""
        lines = stream_lines(
            self._settings,
            path=self._chat_path(),
            body=self._request_body(context),
            transport=self._transport,
        )
        for line in lines:
            yield from self._deltas(line)
        yield from self._flush()

    def _chat_path(self) -> str:
        """Choose the endpoint path for the configured base URL.

        Returns:
            The chat completions path, without repeating a version segment the
            base URL already carries.
        """
        base = self._settings.base_url.rstrip("/")
        if base.endswith(_VERSION_SEGMENT):
            return _VERSIONED_CHAT_PATH
        return _CHAT_PATH
