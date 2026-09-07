"""The contract an answering model implements.

Generation is the far side of the hand-off that retrieval ends at. It takes a
:class:`~rag.domain.models.PromptContext` and produces text, and that is all it
does: it does not retrieve, does not rank, and does not know how the context in
front of it was assembled.

That ignorance is deliberate. The context arrives as a rendered prompt with a
structural boundary between application instructions and untrusted document
content, established once by the prompt augmenter. A model client that
re-templated, re-wrapped, or interleaved that text would move the boundary
without owning it, so implementations send ``rendered_text`` verbatim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from rag.domain.models import Answer, AnswerDelta, PromptContext


class GenerationError(RuntimeError):
    """Raised when the model service cannot produce an answer.

    Messages name the service and the model so that a misconfiguration can be
    corrected, but never quote the prompt or the answer. The prompt contains
    retrieved document content, and an exception is the one place that content
    would otherwise reach logs and tracebacks unredacted.
    """


class BaseChatModel(ABC):
    """Produces an answer from assembled retrieval context.

    Implementations are constructed with whatever they need to reach a model
    and are then reusable across requests. They are synchronous: a caller that
    needs concurrency runs them in a thread, which keeps this package free of
    the async colouring that would otherwise spread through every signature.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identify the model that answers.

        Returns:
            The model's identifier, recorded alongside answers so that a
            result can be attributed to what produced it.
        """

    @abstractmethod
    def stream(self, context: PromptContext) -> Iterator[AnswerDelta]:
        """Answer the question in ``context``, a fragment at a time.

        Args:
            context: Assembled retrieval context. Its ``rendered_text`` is
                sent to the model unchanged.

        Yields:
            Fragments of reasoning and answer, in the order the model
            produced them.

        Raises:
            GenerationError: If the model service is unreachable, rejects the
                request, or returns a response that cannot be understood.
        """

    def answer(self, context: PromptContext) -> Answer:
        """Answer the question in ``context``, waiting for the full result.

        Defined once here rather than in each implementation: consuming a
        stream is not a decision an individual model client gets to make
        differently.

        Args:
            context: Assembled retrieval context.

        Returns:
            The completed answer, carrying the prompt's identity so that the
            answer, the template, and the passages stay attributable together.

        Raises:
            GenerationError: If the model service fails at any point.
        """
        reasoning: list[str] = []
        text: list[str] = []
        for delta in self.stream(context):
            (reasoning if delta.reasoning else text).append(delta.text)
        return Answer(
            text="".join(text),
            reasoning="".join(reasoning),
            model_name=self.model_name,
            prompt_metadata=context.prompt_metadata,
        )
