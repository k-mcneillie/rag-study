"""``create_app`` builds an app with the four routes mounted."""

from __future__ import annotations

from service.app import create_app


def test_create_app_mounts_every_route() -> None:
    """The factory wires health, answer, feedback, and documents.

    The lifespan handler is not run here, so no settings, model, or database
    are touched — only the routing table is checked.
    """
    app = create_app()

    paths = set(app.openapi()["paths"])
    assert {"/health", "/answer", "/feedback", "/documents"} <= paths
