"""Database engine and session construction.

Kept separate from the ORM models and the repository so that connection
concerns—pooling, echo, lifecycle—live in exactly one place and can be
configured per environment without touching persistence logic.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.config import DatabaseSettings


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
        settings.url,
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
