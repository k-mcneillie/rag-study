"""Persistence infrastructure shared by both pipelines.

This package is the only one that imports SQLAlchemy. Processing components
depend on the :class:`~rag.storage.repository.Repository` protocol instead, so
the database can be swapped without touching them.
"""

from rag.storage.engine import build_engine, build_session_factory
from rag.storage.mariadb_repository import MariaDBRepository
from rag.storage.repository import ALLOWED_SEARCH_FILTERS, Repository

__all__ = [
    "ALLOWED_SEARCH_FILTERS",
    "MariaDBRepository",
    "Repository",
    "build_engine",
    "build_session_factory",
]
