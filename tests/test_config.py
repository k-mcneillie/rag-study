"""Tests for environment-driven configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import ConfigurationError, Settings

VALID_ENV = {
    "RAG_DB_HOST": "127.0.0.1",
    "RAG_DB_PORT": "3306",
    "RAG_DB_USER": "rag_app",
    "RAG_DB_PASSWORD": "s3cr3t-value",
    "RAG_DB_NAME": "rag_study",
    "RAG_EMBEDDING_MODEL_PATH": "models/embeddings/all-MiniLM-L6-v2",
    "RAG_EMBEDDING_MODEL_NAME": "all-MiniLM-L6-v2",
}


def test_settings_are_read_from_the_supplied_mapping() -> None:
    """Configuration comes from an injected mapping, not global state."""
    settings = Settings.from_env(VALID_ENV)

    assert settings.database.host == "127.0.0.1"
    assert settings.database.port == 3306
    assert settings.embedding_model_path == Path("models/embeddings/all-MiniLM-L6-v2")
    assert settings.embedding_dimension == 384
    assert settings.max_top_k == 100


def test_missing_required_variable_names_it_without_leaking_values() -> None:
    """A missing variable is reported by name, never by value."""
    env = {key: value for key, value in VALID_ENV.items() if key != "RAG_DB_PASSWORD"}

    with pytest.raises(ConfigurationError) as error:
        Settings.from_env(env)

    assert "RAG_DB_PASSWORD" in str(error.value)


def test_blank_required_variable_is_rejected() -> None:
    """An empty value is treated as missing rather than accepted."""
    with pytest.raises(ConfigurationError):
        Settings.from_env({**VALID_ENV, "RAG_DB_USER": "   "})


def test_non_integer_port_is_rejected_clearly() -> None:
    """A malformed integer fails with a message naming the variable."""
    with pytest.raises(ConfigurationError) as error:
        Settings.from_env({**VALID_ENV, "RAG_DB_PORT": "not-a-port"})

    assert "RAG_DB_PORT" in str(error.value)


def test_password_is_absent_from_settings_repr() -> None:
    """The password cannot reach logs or tracebacks through repr."""
    settings = Settings.from_env(VALID_ENV)

    assert "s3cr3t-value" not in repr(settings.database)
    assert "s3cr3t-value" not in repr(settings)


def test_test_database_key_can_be_targeted() -> None:
    """The integration suite can point at a different schema."""
    env = {**VALID_ENV, "RAG_TEST_DB_NAME": "rag_study_test"}

    settings = Settings.from_env(env, database_key="TEST_DB_NAME")

    assert settings.database.database == "rag_study_test"


def test_prompt_version_is_configuration_not_code() -> None:
    """Changing the prompt in use does not require editing the package.

    The specification forbids hard-coding prompt versions, so that a past
    result can be reproduced by restoring the configuration that produced it.
    """
    settings = Settings.from_env({**VALID_ENV, "RAG_PROMPT_VERSION": "v2"})

    assert settings.prompt_version == "v2"
    assert settings.prompt_name == "retrieval_context"


def test_prompt_identity_falls_back_to_the_shipped_default() -> None:
    """An unconfigured deployment still renders a known prompt."""
    settings = Settings.from_env(VALID_ENV)

    assert (settings.prompt_name, settings.prompt_version) == (
        "retrieval_context",
        "v1",
    )
