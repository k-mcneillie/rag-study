"""Configuration loaded from environment variables.

Nothing that varies between machines or that constitutes a secret is hard-coded
in this package: database credentials, model paths, and operational limits are
all read from the environment. See ``.env.example`` for the recognised
variables.

Settings are read through an injected mapping rather than reaching directly
into :data:`os.environ`, which keeps configuration trivially testable and makes
the set of consumed variables explicit.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

#: Prefix applied to every environment variable this package reads.
ENV_PREFIX = "RAG_"


class ConfigurationError(RuntimeError):
    """Raised when configuration is missing or invalid.

    Error messages name the offending variable but never include its value,
    so that a misconfigured secret cannot leak into logs or tracebacks.
    """


def _require(env: Mapping[str, str], name: str) -> str:
    """Read a required environment variable.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.

    Returns:
        The variable's value.

    Raises:
        ConfigurationError: If the variable is absent or empty.
    """
    key = f"{ENV_PREFIX}{name}"
    value = env.get(key, "").strip()
    if not value:
        raise ConfigurationError(
            f"Required environment variable {key} is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def _optional_str(env: Mapping[str, str], name: str, default: str) -> str:
    """Read an optional string environment variable.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.
        default: Value to use when the variable is absent.

    Returns:
        The value, or ``default`` if the variable is unset or blank.
    """
    return env.get(f"{ENV_PREFIX}{name}", "").strip() or default


def _optional_path(env: Mapping[str, str], name: str) -> Path | None:
    """Read an optional filesystem path from the environment.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.

    Returns:
        The path, or ``None`` if the variable is unset. An unset path means
        the feature it configures is simply not used, never that a default
        location is searched.
    """
    raw = env.get(f"{ENV_PREFIX}{name}", "").strip()
    return Path(raw) if raw else None


def _optional_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Read an optional integer environment variable.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.
        default: Value to use when the variable is absent.

    Returns:
        The parsed integer, or ``default`` if the variable is unset.

    Raises:
        ConfigurationError: If the variable is set but is not an integer.
    """
    key = f"{ENV_PREFIX}{name}"
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {key} must be an integer."
        ) from exc


def _optional_float(env: Mapping[str, str], name: str, default: float) -> float:
    """Read an optional floating-point environment variable.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.
        default: Value to use when the variable is absent.

    Returns:
        The parsed number, or ``default`` if the variable is unset.

    Raises:
        ConfigurationError: If the variable is set but is not a number.
    """
    key = f"{ENV_PREFIX}{name}"
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {key} must be a number."
        ) from exc


def _optional_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Read an optional boolean environment variable.

    Args:
        env: Mapping of environment variable names to values.
        name: Variable name to read, without the ``RAG_`` prefix.
        default: Value to use when the variable is absent.

    Returns:
        The parsed flag, or ``default`` if the variable is unset.

    Raises:
        ConfigurationError: If the variable is set to something other than a
            recognised truth value. An unrecognised value is an error rather
            than a silent ``False``, so a typo cannot quietly disable a
            feature that was meant to be on.
    """
    key = f"{ENV_PREFIX}{name}"
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise ConfigurationError(f"Environment variable {key} must be true or false.")


@dataclass(frozen=True)
class DatabaseSettings:
    """Connection settings for the MariaDB instance.

    Attributes:
        host: Database host name or address.
        port: Database port.
        user: Application database user. This should be a least-privilege
            account, never ``root``.
        password: The user's password. Excluded from ``repr`` so that it
            cannot reach logs or tracebacks through routine object printing.
        database: Name of the schema to connect to.
    """

    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, database_key: str = "DB_NAME"
    ) -> DatabaseSettings:
        """Build database settings from environment variables.

        Args:
            env: Mapping of environment variable names to values.
            database_key: Which variable supplies the schema name. Allows the
                integration suite to target ``RAG_TEST_DB_NAME`` instead of
                the runtime database.

        Returns:
            The parsed database settings.

        Raises:
            ConfigurationError: If a required variable is missing or invalid.
        """
        return cls(
            host=_require(env, "DB_HOST"),
            port=_optional_int(env, "DB_PORT", 3306),
            user=_require(env, "DB_USER"),
            password=_require(env, "DB_PASSWORD"),
            database=_require(env, database_key),
        )


@dataclass(frozen=True)
class GenerationSettings:
    """Connection and sampling settings for the answering model.

    The model is reached over HTTP rather than loaded in-process, so it is a
    service like the database: identified by an address, a credential, and a
    model name. No field here names a particular vendor, and neither does the
    value of ``provider`` except as the name of the client to use, so moving
    from a local server to a hosted one is an environment change.

    Attributes:
        provider: Which client speaks to the service. The entry points map
            this name to an implementation; it selects a protocol, not a
            company.
        base_url: Root URL of the model service.
        model: Which model the service should answer with. It must already be
            available to that service; nothing is downloaded at runtime.
        api_key: Credential sent as a bearer token, or empty when the service
            needs none, as a local server usually does. Excluded from ``repr``
            so it cannot reach logs or tracebacks through routine printing.
        timeout_seconds: How long to wait for the service. Generation on a
            local model is slow enough that the default is generous.
        temperature: Sampling temperature. Low by default, because an answer
            grounded in retrieved passages should follow those passages rather
            than embellish them.
        num_ctx: Context window the service should allocate, in tokens. It
            must comfortably exceed the rendered prompt, or the retrieved
            passages will be silently truncated before the model sees them.
            Services that size their own context ignore it.
        thinking: Whether to ask the model to expose its reasoning separately
            from its answer. Models without the capability ignore it.
    """

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:14b"
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 120.0
    temperature: float = 0.2
    num_ctx: int = 8192
    thinking: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> GenerationSettings:
        """Build generation settings from environment variables.

        Args:
            env: Mapping of environment variable names to values.

        Returns:
            The parsed generation settings.

        Raises:
            ConfigurationError: If a variable is set but invalid.
        """
        return cls(
            provider=_optional_str(env, "LLM_PROVIDER", "ollama").lower(),
            base_url=_optional_str(env, "LLM_BASE_URL", "http://localhost:11434"),
            model=_optional_str(env, "LLM_MODEL", "deepseek-r1:14b"),
            api_key=_optional_str(env, "LLM_API_KEY", ""),
            timeout_seconds=_optional_float(env, "LLM_TIMEOUT_SECONDS", 120.0),
            temperature=_optional_float(env, "LLM_TEMPERATURE", 0.2),
            num_ctx=_optional_int(env, "LLM_NUM_CTX", 8192),
            thinking=_optional_bool(env, "LLM_THINKING", True),
        )


@dataclass(frozen=True)
class Settings:
    """Top-level application settings.

    Attributes:
        database: MariaDB connection settings.
        embedding_model_path: Local directory holding the embedding model's
            weights. Model assets are provisioned separately from Python
            dependencies and are never downloaded at runtime.
        embedding_model_name: Identity of the embedding model, recorded
            alongside every stored vector for reproducibility.
        embedding_dimension: Dimensionality of the embedding model's output.
            Must match the ``VECTOR`` column width in the database schema.
        generation: Settings for the model that answers from retrieved
            context. Generation is optional to the package: retrieval works,
            and is tested, with no model service configured at all.
        max_top_k: Upper bound on the number of results a retrieval request
            may ask for, bounding the cost of any single query.
        top_k: How many passages a request asks for by default. Interfaces
            that do not offer the user a choice use this.
        feedback_path: File that human feedback is appended to. It holds
            question and answer text, so it is kept outside version control.
        embedding_batch_size: How many chunks are encoded at once, bounding
            peak memory when embedding a large document.
        reranker_model_path: Local directory holding the cross-encoder used
            for reranking, or ``None`` to keep the initial ranking.
        reranker_batch_size: How many query/passage pairs are scored at once.
        prompt_name: Logical name of the prompt template to render.
        prompt_version: Version of that template. Changing the prompt in use
            is a configuration change, not a code change, so that a result can
            be reproduced by restoring the configuration that produced it.
        max_document_bytes: Largest source document that will be opened.
        max_document_pages: Most pages extracted from one document.
        chunk_size: Target maximum chunk size, in characters.
        chunk_overlap: Characters shared between adjacent chunks.
        semantic_chunking: Whether to additionally split oversized structural
            sections along sentence-embedding similarity boundaries, rather
            than by character count alone. Off by default: it costs an extra
            pass over the embedding model at ingestion time.
    """

    database: DatabaseSettings
    embedding_model_path: Path
    embedding_model_name: str
    embedding_dimension: int
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    max_top_k: int = 100
    top_k: int = 5
    feedback_path: Path = Path("results/feedback.jsonl")
    embedding_batch_size: int = 32
    reranker_model_path: Path | None = None
    reranker_batch_size: int = 16
    prompt_name: str = "retrieval_context"
    prompt_version: str = "v1"
    max_document_bytes: int = 100 * 1024 * 1024
    max_document_pages: int = 2000
    chunk_size: int = 1200
    chunk_overlap: int = 150
    semantic_chunking: bool = False

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, database_key: str = "DB_NAME"
    ) -> Settings:
        """Build application settings from environment variables.

        Args:
            env: Mapping of environment variable names to values.
            database_key: Which variable supplies the schema name.

        Returns:
            The parsed settings.

        Raises:
            ConfigurationError: If a required variable is missing or invalid.
        """
        return cls(
            database=DatabaseSettings.from_env(env, database_key=database_key),
            embedding_model_path=Path(_require(env, "EMBEDDING_MODEL_PATH")),
            embedding_model_name=_require(env, "EMBEDDING_MODEL_NAME"),
            embedding_dimension=_optional_int(env, "EMBEDDING_DIMENSION", 384),
            generation=GenerationSettings.from_env(env),
            max_top_k=_optional_int(env, "MAX_TOP_K", 100),
            top_k=_optional_int(env, "TOP_K", 5),
            feedback_path=_optional_path(env, "FEEDBACK_PATH")
            or Path("results/feedback.jsonl"),
            embedding_batch_size=_optional_int(env, "EMBEDDING_BATCH_SIZE", 32),
            reranker_model_path=_optional_path(env, "RERANKER_MODEL_PATH"),
            reranker_batch_size=_optional_int(env, "RERANKER_BATCH_SIZE", 16),
            prompt_name=_optional_str(env, "PROMPT_NAME", "retrieval_context"),
            prompt_version=_optional_str(env, "PROMPT_VERSION", "v1"),
            max_document_bytes=_optional_int(
                env, "MAX_DOCUMENT_BYTES", 100 * 1024 * 1024
            ),
            max_document_pages=_optional_int(env, "MAX_DOCUMENT_PAGES", 2000),
            chunk_size=_optional_int(env, "CHUNK_SIZE", 1200),
            chunk_overlap=_optional_int(env, "CHUNK_OVERLAP", 150),
            semantic_chunking=_optional_bool(env, "SEMANTIC_CHUNKING", False),
        )


def load_settings(
    *, dotenv_path: Path | None = None, database_key: str = "DB_NAME"
) -> Settings:
    """Load settings, populating the environment from a ``.env`` file first.

    Args:
        dotenv_path: Explicit path to a ``.env`` file. When omitted, the file
            is discovered by walking up from the current working directory.
        database_key: Which variable supplies the schema name.

    Returns:
        The parsed settings.

    Raises:
        ConfigurationError: If a required variable is missing or invalid.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return Settings.from_env(os.environ, database_key=database_key)
