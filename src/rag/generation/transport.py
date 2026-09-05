"""HTTP plumbing shared by every model client.

Reaching a model service means the same four things whatever the service is:
authenticate, post a request, refuse to continue on an error status, and read
a streamed body line by line. Only the shape of the JSON differs between
providers, so only that belongs in a provider's own module.

Keeping the plumbing here also keeps one security property in one place. The
request body contains retrieved document content, and the headers contain a
credential; neither is ever quoted in an exception. Failures name the service,
the model, and the status, which is what a reader needs and all they need.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import httpx

from rag.config import GenerationSettings
from rag.generation.interfaces import GenerationError


def auth_headers(settings: GenerationSettings) -> dict[str, str]:
    """Build the request headers for a service.

    Args:
        settings: Settings that may carry a credential.

    Returns:
        Headers including a bearer token when one is configured. A local
        service that needs no credential is left unauthenticated rather than
        being sent an empty one.
    """
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    return headers


def stream_lines(
    settings: GenerationSettings,
    *,
    path: str,
    body: Mapping[str, Any],
    transport: httpx.BaseTransport | None = None,
) -> Iterator[str]:
    """Post a request and yield the non-empty lines of the streamed response.

    Args:
        settings: Where the service is, and how long to wait for it.
        path: Endpoint path, appended to the configured base URL.
        body: The JSON body to post.
        transport: An HTTP transport to use instead of the default. Only tests
            supply this, so a client can be exercised without a server.

    Yields:
        Each non-blank line of the response body, in order.

    Raises:
        GenerationError: If the service is unreachable, times out, or returns
            an error status.
    """
    url = f"{settings.base_url.rstrip('/')}{path}"
    try:
        with (
            httpx.Client(
                timeout=settings.timeout_seconds, transport=transport
            ) as client,
            client.stream(
                "POST", url, json=dict(body), headers=auth_headers(settings)
            ) as response,
        ):
            _check_status(response, settings)
            for line in response.iter_lines():
                if line.strip():
                    yield line
    except httpx.HTTPError as exc:
        raise GenerationError(
            f"Cannot reach the model service at {settings.base_url}. Check that "
            f"it is running and that model {settings.model} is available there."
        ) from exc


def _check_status(response: httpx.Response, settings: GenerationSettings) -> None:
    """Fail early on a rejected request.

    Args:
        response: The streaming response, before its body is consumed.
        settings: Settings naming the service and model, for the message.

    Raises:
        GenerationError: If the service returned an error status. A 404 and an
            authentication failure are called out separately: they are the two
            failures a reader can act on immediately, and they are easily
            confused with each other from a bare status code.
    """
    if response.is_success:
        return

    response.read()
    if response.status_code == httpx.codes.NOT_FOUND:
        raise GenerationError(
            f"Model {settings.model} is not available at {settings.base_url}. "
            f"Provision it there first; it is never downloaded automatically."
        )
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        raise GenerationError(
            f"The model service at {settings.base_url} rejected the credential "
            f"(HTTP {response.status_code}). Check RAG_LLM_API_KEY."
        )
    raise GenerationError(
        f"The model service at {settings.base_url} rejected the request for "
        f"model {settings.model} (HTTP {response.status_code})."
    )
