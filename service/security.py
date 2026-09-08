"""The service's only access control: an optional shared API key.

If ``RAG_API_KEY`` is set, every route except ``GET /health`` requires it in
the ``X-API-Key`` header, compared with :func:`hmac.compare_digest` so a wrong
key is rejected in the same time a right one is accepted. If it is **not** set,
the service runs open — which is convenient for local development and a
deliberate footgun anywhere else. The startup log says which mode is active.

There is no per-user identity, no rotation, and no rate limiting; those, along
with TLS, belong to a reverse proxy in front of the service and are listed in
``docs/future-work.md``. The pattern mirrors ``RAG_LLM_API_KEY`` on the
generation side: a secret that lives only in ``.env``.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

#: Environment variable holding the shared secret.
API_KEY_ENV = "RAG_API_KEY"

#: Header the secret is expected in.
API_KEY_HEADER = "X-API-Key"


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject a request that does not carry the configured API key.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header, if present.

    Raises:
        HTTPException: With status 401 when ``RAG_API_KEY`` is set and the
            header is absent or does not match. When ``RAG_API_KEY`` is unset,
            every request is allowed.
    """
    expected = os.environ.get(API_KEY_ENV, "").strip()
    if not expected:
        return
    provided = x_api_key or ""
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )
