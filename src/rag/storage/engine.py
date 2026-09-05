"""Database engine and session construction.

Kept separate from the ORM models and the repository so that connection
concerns—pooling, echo, lifecycle—live in exactly one place and can be
configured per environment without touching persistence logic.
"""

from __future__ import annotations

from sqlalchemy import URL, Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.config import DatabaseSettings


def build_url(settings: DatabaseSettings) -> URL:
    """Build the SQLAlchemy connection URL for a database.

    URL construction lives here rather than on the settings object so that
    configuration stays free of SQLAlchemy: the settings are plain data, and
    only this layer knows how to reach a database.

    ``URL.create`` escapes credentials correctly, so a password containing
    reserved characters cannot corrupt the URL, and the object it returns
    masks the password in its own ``repr``.

    Args:
        settings: Connection settings for the MariaDB instance.

    Returns:
        A SQLAlchemy URL for the configured database.
    """
    return URL.create(
        drivername="mysql+pymysql",
        username=settings.user,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        database=settings.database,
        query={"charset": "utf8mb4"},
    )


def build_engine(settings: DatabaseSettings, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for the configured database.

    Args:
        settings: Connection settings for the MariaDB instance.
        echo: Whether to log every SQL statement. Off by default, because
            echoed statements would place document text and query parameters
            into the logs.

    Returns:
        A configured SQLAlchemy engine.
    """
    return create_engine(
        build_url(settings),
        echo=echo,
        pool_pre_ping=True,
        future=True,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to an engine.

    Args:
        engine: The engine sessions should be bound to.

    Returns:
        A session factory producing sessions for that engine.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)
