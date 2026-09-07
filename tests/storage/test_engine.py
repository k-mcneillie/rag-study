"""Tests for connection URL construction.

URL building lives in the storage layer rather than on the settings object, so
that configuration stays plain data with no SQLAlchemy dependency.
"""

from __future__ import annotations

from rag.config import DatabaseSettings
from rag.storage.engine import build_url

SETTINGS = DatabaseSettings(
    host="127.0.0.1",
    port=3306,
    user="rag_app",
    password="s3cr3t-value",
    database="rag_study",
)


def test_url_targets_the_configured_database() -> None:
    """The URL addresses the configured host, port, and schema."""
    rendered = build_url(SETTINGS).render_as_string(hide_password=False)

    assert rendered.startswith("mysql+pymysql://rag_app:")
    assert rendered.endswith("@127.0.0.1:3306/rag_study?charset=utf8mb4")


def test_reserved_characters_in_a_password_are_escaped() -> None:
    """A password containing URL metacharacters cannot corrupt the URL."""
    settings = DatabaseSettings(
        host="127.0.0.1",
        port=3306,
        user="rag_app",
        password="p@ss:word/with?reserved#chars",
        database="rag_study",
    )

    rendered = build_url(settings).render_as_string(hide_password=False)

    assert "p%40ss%3Aword%2Fwith%3Freserved%23chars" in rendered
    assert rendered.endswith("/rag_study?charset=utf8mb4")


def test_password_is_masked_when_a_url_is_printed() -> None:
    """Printing a URL does not disclose the password."""
    assert "s3cr3t-value" not in repr(build_url(SETTINGS))
    assert "s3cr3t-value" not in str(build_url(SETTINGS))
