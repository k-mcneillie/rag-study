"""Create the database schema. An administrative operation, run by hand.

The application user deliberately holds no DDL privileges, so it cannot create
its own tables: an account that can only read and write rows cannot be used to
alter the schema, whatever a bug or an injected statement might attempt. That
separation means schema creation needs a separate, privileged connection, which
is what this script provides.

Run it once before the first ingestion, and again after any schema change,
with ``RAG_ADMIN_DB_URL`` set to a SQLAlchemy URL for a privileged account.
See the README for a worked example.

The URL is read from the environment rather than accepted as an argument, so
administrative credentials do not end up in shell history or process listings.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine

from rag.storage.orm import Base

ADMIN_URL_VAR = "RAG_ADMIN_DB_URL"


def main() -> int:
    """Create every table the package needs.

    Returns:
        A process exit status: ``0`` on success, ``1`` if the administrative
        connection URL is not configured.
    """
    url = os.environ.get(ADMIN_URL_VAR, "").strip()
    if not url:
        print(
            f"{ADMIN_URL_VAR} is not set. Provide a SQLAlchemy URL for an "
            f"account holding DDL privileges on the target database.",
            file=sys.stderr,
        )
        return 1

    engine = create_engine(url)
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    tables = ", ".join(sorted(Base.metadata.tables))
    print(f"Schema created. Tables: {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
