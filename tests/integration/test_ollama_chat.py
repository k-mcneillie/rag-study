"""End-to-end generation against a real model service.

The unit tests prove the client speaks the protocol correctly. This proves the
protocol is the one the installed server actually implements — the assumption
a mock transport cannot check, and the one that breaks when a dependency moves
underneath the code.

It skips, rather than fails, when no server is running or the configured model
is absent, so the suite stays runnable on a machine without either.
"""

from __future__ import annotations

import httpx
import pytest

from rag.config import GenerationSettings, load_settings
from rag.domain.models import (
    Chunk,
    PromptContext,
    PromptMetadata,
    RankedChunk,
    RerankedChunk,
)
from rag.generation import GenerationError, OllamaChatModel

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def generation_settings() -> GenerationSettings:
    """Provide settings pointed at a reachable server holding the model.

    Returns:
        The configured generation settings.
    """
    try:
        settings = load_settings().generation
    except Exception as exc:
        pytest.skip(f"Generation configuration unavailable: {exc}")

    try:
        response = httpx.get(f"{settings.base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.skip(f"Model service is not reachable: {type(exc).__name__}")

    available = {model["name"] for model in response.json().get("models", [])}
    if settings.model not in available:
        pytest.skip(f"Model {settings.model} is not provisioned on the server.")
    return settings


def _context() -> PromptContext:
    """Build a small, self-contained prompt with one unambiguous answer.

    The passage carries a fact no model could know otherwise, so an answer
    that contains it demonstrates the retrieved text actually reached the
    model rather than being answered from training data.

    Returns:
        A prompt context ready to send.
    """
    fact = "The Kestrel-7 telescope was calibrated on 4 March 2019."
    chunk = Chunk(document_id="doc-1", page_number=1, text=fact, section="1 Instrument")
    reranked = RerankedChunk(
        ranked_chunk=RankedChunk(chunk=chunk, score=0.9, rank=0),
        rerank_score=0.9,
        rank=0,
    )
    rendered = (
        "Answer only from the source material below, and cite it as [1].\n\n"
        "===== BEGIN SOURCE MATERIAL =====\n"
        f"[1] (document=doc-1 page=1)\n{fact}\n"
        "===== END SOURCE MATERIAL =====\n\n"
        "QUESTION: On what date was the Kestrel-7 telescope calibrated?"
    )
    return PromptContext(
        chunks=(reranked,),
        rendered_text=rendered,
        prompt_metadata=PromptMetadata("retrieval_context", "v1"),
    )


def test_the_server_answers_from_the_supplied_passage(
    generation_settings: GenerationSettings,
) -> None:
    """A real server returns an answer grounded in the passage it was given.

    Args:
        generation_settings: Settings for a reachable server and model.
    """
    model = OllamaChatModel(generation_settings)

    answer = model.answer(_context())

    assert answer.text.strip()
    assert "2019" in answer.text
    assert answer.model_name == generation_settings.model


def test_reasoning_arrives_separately_from_the_answer(
    generation_settings: GenerationSettings,
) -> None:
    """The server keeps reasoning out of the answer text.

    This is the reason the native endpoint is used rather than the
    OpenAI-compatible one. If it ever stops holding, the answer shown to a
    reader would start with the model's private deliberation.

    Args:
        generation_settings: Settings for a reachable server and model.
    """
    model = OllamaChatModel(generation_settings)

    answer = model.answer(_context())

    assert "<think>" not in answer.text
    assert "</think>" not in answer.text


def test_a_missing_model_is_reported_not_downloaded(
    generation_settings: GenerationSettings,
) -> None:
    """Asking for an unprovisioned model fails; it never pulls it.

    Args:
        generation_settings: Settings for a reachable server.
    """
    model = OllamaChatModel(
        GenerationSettings(
            base_url=generation_settings.base_url,
            model="not-a-real-model:0b",
            timeout_seconds=10.0,
        )
    )

    with pytest.raises(GenerationError):
        list(model.stream(_context()))
