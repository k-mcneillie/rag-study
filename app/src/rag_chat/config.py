"""Configuration for the chat app, read from the environment.

The app needs almost nothing: where the API service is, the key to reach it,
how many passages to ask for, and whether to run in demo mode. A ``.env`` file
beside this directory is loaded if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

#: Recognised truthy spellings for ``RAG_CHAT_DEMO``.
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Everything the app reads from its environment.

    Attributes:
        api_url: Base URL of the rag-study API service.
        api_key: Shared secret sent as ``X-API-Key``. May be empty when the
            service was started without ``RAG_API_KEY`` (it then runs open), or
            when running purely in demo mode.
        top_k: How many passages to request per question. The service clamps
            this to its own maximum.
        demo: Force demo mode even when a service is reachable. Demo mode is
            also entered automatically when the service cannot be reached.
    """

    api_url: str
    api_key: str
    top_k: int
    demo: bool

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build the configuration, loading a ``.env`` file first.

        Returns:
            The parsed configuration.
        """
        load_dotenv()
        raw_top_k = os.environ.get("RAG_CHAT_TOP_K", "5").strip()
        return cls(
            api_url=os.environ.get("RAG_API_URL", "http://localhost:8080").strip(),
            api_key=os.environ.get("RAG_API_KEY", "").strip(),
            top_k=int(raw_top_k) if raw_top_k else 5,
            demo=os.environ.get("RAG_CHAT_DEMO", "").strip().lower() in _TRUTHY,
        )
