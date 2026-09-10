"""Configuration for the chat app, read from the environment.

The app needs where the provider is, the key to reach it, which model to name in
the request, where to send ratings, and whether to force demo mode. A ``.env``
file beside this directory is loaded if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

#: Values that read as "on" for the boolean environment variables.
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """The app's runtime configuration.

    Attributes:
        base_url: Root of the OpenAI-language provider, with or without a
            trailing ``/v1``.
        api_key: Bearer token for the provider. May be empty — for a local
            server that needs no credential, or in demo mode.
        model: Model id to name in the request body. Empty means "resolve it
            from ``GET /v1/models`` at start".
        feedback_url: Optional endpoint for the rating buttons. Empty means
            ratings are acknowledged but not stored.
        demo: Force demo mode. Demo mode is also entered automatically when the
            provider cannot be reached.
    """

    base_url: str
    api_key: str
    model: str
    feedback_url: str
    demo: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        """Read the configuration from ``RAG_OPENAI_*`` environment variables.

        Returns:
            The parsed configuration.
        """
        load_dotenv()
        return cls(
            base_url=os.environ.get(
                "RAG_OPENAI_BASE_URL", "https://api.openai.com"
            ).strip(),
            api_key=os.environ.get("RAG_OPENAI_API_KEY", "").strip(),
            model=os.environ.get("RAG_OPENAI_MODEL", "").strip(),
            feedback_url=os.environ.get("RAG_OPENAI_FEEDBACK_URL", "").strip(),
            demo=os.environ.get("RAG_OPENAI_DEMO", "").strip().lower() in _TRUTHY,
        )
