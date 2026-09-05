"""Choosing a model client from configuration.

The package's standing position is that constructor injection is enough and
that a registry would be indirection bought before it was needed. This is the
one place that does not hold, and for a reason worth stating: which model
service answers is a deployment decision, not a code decision. If selecting a
provider required editing an entry point, then "point it at another service"
would mean a code change in every entry point, and the swappability the
interface provides would exist only on paper.

So this is deliberately the smallest thing that closes that gap: a mapping
from a configured name to a class, and an error naming the alternatives. It
does not discover implementations, does not import anything lazily, and has no
opinion about anything but chat models. Adding a provider means writing a
:class:`~rag.generation.interfaces.BaseChatModel` and adding one line here.
"""

from __future__ import annotations

from collections.abc import Callable

from rag.config import GenerationSettings
from rag.generation.interfaces import BaseChatModel, GenerationError
from rag.generation.ollama import OllamaChatModel
from rag.generation.openai_compatible import OpenAICompatibleChatModel

#: Configured provider name to the client that speaks that protocol. The names
#: describe protocols, not companies, and several map to the same client:
#: ``vllm`` and ``openai`` are the same wire format, kept as separate names
#: because a reader configuring vLLM should find vLLM in the list rather than
#: having to know that it is OpenAI-compatible.
CHAT_MODELS: dict[str, Callable[[GenerationSettings], BaseChatModel]] = {
    "ollama": OllamaChatModel,
    "openai": OpenAICompatibleChatModel,
    "vllm": OpenAICompatibleChatModel,
}


def chat_model_for(settings: GenerationSettings) -> BaseChatModel:
    """Build the model client the configuration asks for.

    Args:
        settings: Generation settings naming a provider.

    Returns:
        A client bound to the configured service.

    Raises:
        GenerationError: If the configured provider is not one this package
            implements. The message lists what is available, because the
            alternative is a deployment that starts and then fails on the
            first question.
    """
    try:
        build = CHAT_MODELS[settings.provider]
    except KeyError:
        known = ", ".join(sorted(CHAT_MODELS))
        raise GenerationError(
            f"Unknown model provider {settings.provider!r}. "
            f"RAG_LLM_PROVIDER must be one of: {known}."
        ) from None
    return build(settings)
