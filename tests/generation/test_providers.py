"""Tests for choosing a model client from configuration."""

from __future__ import annotations

import pytest

from rag.config import GenerationSettings
from rag.generation import (
    CHAT_MODELS,
    BaseChatModel,
    GenerationError,
    OllamaChatModel,
    OpenAICompatibleChatModel,
    chat_model_for,
)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("ollama", OllamaChatModel),
        ("openai", OpenAICompatibleChatModel),
        ("vllm", OpenAICompatibleChatModel),
    ],
)
def test_the_configured_provider_selects_its_client(
    provider: str, expected: type[BaseChatModel]
) -> None:
    """Changing service is a configuration change, not a code change.

    Args:
        provider: The configured provider name.
        expected: The client that name must resolve to.
    """
    model = chat_model_for(GenerationSettings(provider=provider))

    assert isinstance(model, expected)


def test_every_registered_provider_implements_the_interface() -> None:
    """Nothing can be registered that the rest of the system cannot use."""
    for provider in CHAT_MODELS:
        model = chat_model_for(GenerationSettings(provider=provider))
        assert isinstance(model, BaseChatModel)


def test_an_unknown_provider_fails_at_startup_naming_the_options() -> None:
    """A misconfiguration is caught when the client is built.

    The alternative is a deployment that starts cleanly and then fails on the
    reader's first question, which is a far worse place to discover a typo.
    """
    with pytest.raises(GenerationError) as caught:
        chat_model_for(GenerationSettings(provider="gpt-9"))

    message = str(caught.value)
    assert "gpt-9" in message
    assert "ollama" in message
    assert "vllm" in message


def test_the_configured_model_reaches_the_client() -> None:
    """The selected client answers with the model that was configured."""
    model = chat_model_for(GenerationSettings(provider="vllm", model="Qwen/Qwen3-8B"))

    assert model.model_name == "Qwen/Qwen3-8B"
