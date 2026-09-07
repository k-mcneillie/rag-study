"""Create or check the database schema. An administrative operation.

The application user deliberately holds no DDL privileges, so it cannot create
or alter its own tables: an account that can only read and write rows cannot
change the schema, whatever a bug or an injected statement might attempt. That
separation means schema work needs a separate, privileged connection, which is
what this script provides.

It also reports **drift**. ``CREATE TABLE`` only ever adds missing tables, so a
column added to the models after a database was created is silently absent
until a query fails with an unhelpful "unknown column" error at run time. This
script compares the models against the live database and names the difference,
which is a stopgap until migrations exist — see docs/future-work.md.

Run it once before the first ingestion, and again after any schema change,
with ``RAG_ADMIN_DB_URL`` set to a SQLAlchemy URL for a privileged account.
See the README for a worked example.

The URL is read from the environment rather than accepted as an argument, so
administrative credentials do not end up in shell history or process listings.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import Engine, create_engine, text

from rag.storage.orm import Base

ADMIN_URL_VAR = "RAG_ADMIN_DB_URL"

#: Columns are read from ``information_schema`` rather than through
#: SQLAlchemy's reflection, which cannot parse MariaDB's ``VECTOR`` type and
#: raises while trying. Only column names are needed here, so the simpler
#: query is also the more robust one.
_COLUMNS_QUERY = text(
    "SELECT table_name, column_name FROM information_schema.columns "
    "WHERE table_schema = DATABASE()"
)


def find_drift(engine: Engine) -> dict[str, list[str]]:
    """Find columns the models declare that the database does not have.

    Args:
        engine: An engine connected to the database to inspect.

    Returns:
        Missing column names, keyed by table. Empty when the database matches
        the models. Tables absent entirely are not reported, since creating
        the schema adds them.
    """
    with engine.connect() as connection:
        rows = connection.execute(_COLUMNS_QUERY).all()

    present: dict[str, set[str]] = {}
    for table_name, column_name in rows:
        present.setdefault(str(table_name), set()).add(str(column_name))

    drift: dict[str, list[str]] = {}
    for name, table in Base.metadata.tables.items():
        if name not in present:
            continue
        missing = [
            column.name for column in table.columns if column.name not in present[name]
        ]
        if missing:
            drift[name] = missing
    return drift


def main() -> int:
    """Create any missing tables and report any schema drift.

    Returns:
        A process exit status: ``0`` when the database matches the models,
        ``1`` if the connection URL is unset or drift was found. Drift is an
        error rather than a warning, because the alternative is discovering it
        as a failed query during retrieval.
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
        drift = find_drift(engine)
    finally:
        engine.dispose()

    tables = ", ".join(sorted(Base.metadata.tables))
    print(f"Schema present. Tables: {tables}")

    if drift:
        print(
            "\nThis database predates the current models and is missing "
            "columns. Creating tables cannot add them, so they must be added "
            "by hand until migrations exist:",
            file=sys.stderr,
        )
        for table, columns in sorted(drift.items()):
            for column in columns:
                declared = Base.metadata.tables[table].columns[column]
                print(
                    f"  ALTER TABLE {table} ADD COLUMN {column} "
                    f"{declared.type.compile()};",
                    file=sys.stderr,
                )
        print("\nAlternatively, drop the database and re-ingest.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
