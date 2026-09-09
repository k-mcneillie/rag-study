"""Configuration for the chat app, read from the environment.

The app needs where the provider is, the key to reach it, which model to name in
the request, a few optional retrieval knobs, and whether to run in demo or trace
mode. A ``.env`` file beside this directory is loaded if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

#: Recognised truthy spellings for the boolean variables.
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Everything the app reads from its environment.

    Attributes:
        base_url: Root URL of the OpenAI-language RAG provider, with or without
            a trailing ``/v1``.
        api_key: Secret sent as ``Authorization: Bearer``. May be empty when
            talking to a provider that needs no credential, or in demo mode.
        model: Model id to name in the request body. Empty means "resolve it
            from ``GET /v1/models`` at start".
        vector_store: Optional index id, sent in the request only when the
            provider accepts one. Normally the provider knows its own index.
        top_k: Passage-count hint. Sent only when the provider documents one.
        feedback_url: Optional endpoint for the rating buttons. Empty means
            ratings are acknowledged but not stored.
        trace: Show the "OpenAI wire trace" element for live answers too. It is
            always shown in demo mode.
        demo: Force demo mode even when a provider is reachable. Demo mode is
            also entered automatically when the provider cannot be reached.
    """

    base_url: str
    api_key: str
    model: str
    vector_store: str
    top_k: int
    feedback_url: str
    trace: bool
    demo: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build the configuration, loading a ``.env`` file first.

        Returns:
            The parsed configuration.
        """
        load_dotenv()
        raw_top_k = os.environ.get("RAG_OPENAI_TOP_K", "5").strip()
        return cls(
            base_url=os.environ.get(
                "RAG_OPENAI_BASE_URL", "https://api.openai.com"
            ).strip(),
            api_key=os.environ.get("RAG_OPENAI_API_KEY", "").strip(),
            model=os.environ.get("RAG_OPENAI_MODEL", "").strip(),
            vector_store=os.environ.get("RAG_OPENAI_VECTOR_STORE", "").strip(),
            top_k=int(raw_top_k) if raw_top_k else 5,
            feedback_url=os.environ.get("RAG_OPENAI_FEEDBACK_URL", "").strip(),
            trace=os.environ.get("RAG_OPENAI_TRACE", "").strip().lower() in _TRUTHY,
            demo=os.environ.get("RAG_OPENAI_DEMO", "").strip().lower() in _TRUTHY,
        )
