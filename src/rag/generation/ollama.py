"""An answering model served by Ollama's native API.

Ollama's own ``/api/chat`` endpoint is used rather than its OpenAI-compatible
one for a single concrete reason: models that reason return that reasoning in
a separate ``thinking`` field instead of wrapping it in tags inside the
answer. Splitting a stream on markers the model itself emits would be
guesswork; taking two fields the protocol already separates is not.

It also accepts ``num_ctx``, which matters here: a prompt carrying several
retrieved passages is easily long enough to be truncated by a default context
window, silently and without the passages ever reaching the model.

Everything that is not specific to this protocol lives in
:mod:`rag.generation.transport`.
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

#: The service's streaming chat endpoint, relative to its base URL.
_CHAT_PATH = "/api/chat"


class OllamaChatModel(BaseChatModel):
    """Answers by streaming from an Ollama server.

    The server is not started, managed, or provisioned from here: the model
    must already be present on it. A missing model is an error naming what is
    missing, never an attempt to pull it, which would be a runtime download.
    """

    def __init__(
        self,
        settings: GenerationSettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Prepare a client for an Ollama server.

        Args:
            settings: Where the server is and how it should sample.
            transport: An HTTP transport to use instead of the default. Only
                tests supply this, so that the client can be exercised in full
                without a server and without reaching the network.
        """
        self._settings = settings
        self._transport = transport

    @property
    def model_name(self) -> str:
        """Identify the configured model.

        Returns:
            The model name as configured, for example ``"deepseek-r1:14b"``.
        """
        return self._settings.model

    def _request_body(self, context: PromptContext) -> dict[str, Any]:
        """Build the request payload for one question.

        The rendered prompt is sent as a single user message, exactly as the
        augmenter produced it. It is not split into system and user parts:
        the template's instructions and its source-material markers are one
        structure, and dividing them across roles would weaken it.

        Args:
            context: Assembled retrieval context.

        Returns:
            The JSON body to post.
        """
        return {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": context.rendered_text}],
            "stream": True,
            "think": self._settings.thinking,
            "options": {
                "temperature": self._settings.temperature,
                "num_ctx": self._settings.num_ctx,
            },
        }

    def _deltas(self, line: str) -> Iterator[AnswerDelta]:
        """Interpret one line of the streamed response.

        Args:
            line: A single line of newline-delimited JSON.

        Yields:
            The fragments the line carries: reasoning, answer, or neither.

        Raises:
            GenerationError: If the line is not JSON, or reports an error.
        """
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"{self._settings.base_url} returned a malformed response "
                f"for model {self._settings.model}."
            ) from exc

        if error := payload.get("error"):
            raise GenerationError(
                f"Model {self._settings.model} failed at "
                f"{self._settings.base_url}: {error}"
            )

        message = payload.get("message") or {}
        if reasoning := message.get("thinking"):
            yield AnswerDelta(text=reasoning, reasoning=True)
        if text := message.get("content"):
            yield AnswerDelta(text=text)

    def stream(self, context: PromptContext) -> Iterator[AnswerDelta]:
        """Answer the question in ``context``, a fragment at a time.

        Args:
            context: Assembled retrieval context, sent unchanged.

        Yields:
            Fragments of reasoning and answer, in the order they arrive.

        Raises:
            GenerationError: If the server is unreachable, rejects the
                request, or streams something unintelligible.
        """
        logger.info(
            "Generating with model %s at %s.",
            self._settings.model,
            self._settings.base_url,
        )
        lines = stream_lines(
            self._settings,
            path=_CHAT_PATH,
            body=self._request_body(context),
            transport=self._transport,
        )
        for line in lines:
            yield from self._deltas(line)
