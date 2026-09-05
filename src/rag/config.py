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
from sqlalchemy import URL

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

    @property
    def url(self) -> URL:
        """Build the SQLAlchemy connection URL.

        ``URL.create`` escapes credentials correctly, so passwords containing
        reserved characters cannot corrupt the URL. The returned object masks
        the password in its own ``repr``.

        Returns:
            A SQLAlchemy URL for the configured MariaDB database.
        """
        return URL.create(
            drivername="mysql+pymysql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"charset": "utf8mb4"},
        )

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
        max_top_k: Upper bound on the number of results a retrieval request
            may ask for, bounding the cost of any single query.
    """

    database: DatabaseSettings
    embedding_model_path: Path
    embedding_model_name: str
    embedding_dimension: int
    max_top_k: int = 100

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
            max_top_k=_optional_int(env, "MAX_TOP_K", 100),
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
