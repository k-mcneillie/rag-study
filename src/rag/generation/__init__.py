"""Answering from assembled retrieval context.

This package sits beside the retrieval pipeline rather than inside it. It
consumes the ``PromptContext`` contract and knows nothing else about
retrieval; retrieval, in turn, does not know that it exists. Either can be
tested, replaced, or deleted without touching the other.

Which service answers is configuration, not code: see
:func:`rag.generation.providers.chat_model_for`.
"""

from rag.generation.interfaces import BaseChatModel, GenerationError
from rag.generation.ollama import OllamaChatModel
from rag.generation.openai_compatible import OpenAICompatibleChatModel
from rag.generation.providers import CHAT_MODELS, chat_model_for

__all__ = [
    "CHAT_MODELS",
    "BaseChatModel",
    "GenerationError",
    "OllamaChatModel",
    "OpenAICompatibleChatModel",
    "chat_model_for",
]
