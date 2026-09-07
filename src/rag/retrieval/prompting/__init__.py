"""Assembly of retrieved chunks into prompt context."""

from rag.retrieval.prompting.augmenter import (
    PromptTemplateNotFoundError,
    TemplatePromptAugmenter,
)

__all__ = ["PromptTemplateNotFoundError", "TemplatePromptAugmenter"]
