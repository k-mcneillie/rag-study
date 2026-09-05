"""Tests for the MariaDB ``VECTOR`` column type.

These exercise serialisation and SQL generation without touching a database.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import mysql

from rag.storage.vector import Vector


def test_column_specification_carries_the_dimension() -> None:
    """DDL renders the configured width."""
    assert Vector(384).get_col_spec() == "VECTOR(384)"


def test_zero_or_negative_dimension_is_rejected() -> None:
    """A vector column must have a positive width."""
    with pytest.raises(ValueError, match="positive integer"):
        Vector(0)


def test_vectors_round_trip_through_the_processors() -> None:
    """A vector survives serialisation and parsing unchanged."""
    vector_type = Vector(3)
    serialise = vector_type.bind_processor(mysql.dialect())
    parse = vector_type.result_processor(mysql.dialect(), None)

    assert parse(serialise((0.5, -1.5, 2.0))) == (0.5, -1.5, 2.0)


def test_wrong_width_is_rejected_before_reaching_the_database() -> None:
    """A mismatched vector fails fast rather than corrupting a row."""
    serialise = Vector(3).bind_processor(mysql.dialect())

    with pytest.raises(ValueError, match="3-dimensional"):
        serialise((0.1, 0.2))


def test_bytes_results_are_decoded() -> None:
    """Drivers returning bytes are handled as well as those returning str."""
    parse = Vector(2).result_processor(mysql.dialect(), None)

    assert parse(b"[1.0,2.0]") == (1.0, 2.0)


def test_null_values_pass_through() -> None:
    """A NULL vector is neither serialised nor parsed."""
    vector_type = Vector(2)

    assert vector_type.bind_processor(mysql.dialect())(None) is None
    assert vector_type.result_processor(mysql.dialect(), None)(None) is None


def test_values_are_bound_not_interpolated() -> None:
    """Vectors reach MariaDB through VEC_FromText on a bound parameter."""
    from sqlalchemy import literal, select

    statement = select(literal((1.0, 2.0), Vector(2)).label("v"))
    compiled = statement.compile(dialect=mysql.dialect())

    assert "VEC_FromText" in str(compiled)
    assert "1.0" not in str(compiled)
