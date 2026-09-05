"""Shared pytest fixtures.

Unit tests run with no external dependencies. Integration tests need a live
MariaDB server and are skipped, not failed, when one is unavailable, so the
suite stays runnable on a machine without a database.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from rag.config import ConfigurationError, Settings, load_settings
from rag.storage.engine import build_engine, build_session_factory
from rag.storage.mariadb_repository import MariaDBRepository
from rag.storage.orm import Base


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    """Load settings pointed at the disposable test schema.

    Returns:
        Settings whose database is the integration test schema.
    """
    try:
        return load_settings(database_key="TEST_DB_NAME")
    except ConfigurationError as exc:
        pytest.skip(f"Database configuration unavailable: {exc}")


@pytest.fixture(scope="session")
def engine(integration_settings: Settings) -> Iterator[Engine]:
    """Provide an engine bound to the test schema, skipping if unreachable.

    Args:
        integration_settings: Settings for the test schema.

    Yields:
        An engine connected to the test database.
    """
    engine = build_engine(integration_settings.database)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        engine.dispose()
        pytest.skip(f"MariaDB is not reachable: {type(exc).__name__}")
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> Iterator[sessionmaker[Session]]:
    """Create the schema for one test and drop it afterwards.

    Args:
        engine: Engine bound to the test database.

    Yields:
        A session factory for the freshly created schema.
    """
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield build_session_factory(engine)
    Base.metadata.drop_all(engine)


@pytest.fixture
def repository(
    session_factory: sessionmaker[Session], integration_settings: Settings
) -> MariaDBRepository:
    """Build a repository against the prepared test schema.

    Args:
        session_factory: Factory for sessions on the test database.
        integration_settings: Settings supplying the embedding dimension.

    Returns:
        A repository ready for use in an integration test.
    """
    return MariaDBRepository(
        session_factory,
        embedding_dimension=integration_settings.embedding_dimension,
        max_top_k=integration_settings.max_top_k,
    )
