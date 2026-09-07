"""SQLAlchemy type for MariaDB's native ``VECTOR`` column.

MariaDB 11.7 and later store embeddings in a dedicated ``VECTOR(n)`` column and
compare them with functions such as ``VEC_DISTANCE_COSINE``. Values cross the
wire in MariaDB's own binary format, so this type converts between that format
and plain Python tuples using the server's ``VEC_FromText`` and ``VEC_ToText``
functions.

Conversion happens inside bound parameters rather than by interpolating values
into SQL strings, so a vector can never contribute to SQL injection. Decoding
uses :func:`json.loads`, never :func:`eval` or :mod:`pickle`, in line with the
project's rule against unsafe deserialisation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import Dialect, func
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[tuple[float, ...]]):
    """A MariaDB ``VECTOR(n)`` column mapped to a tuple of floats.

    Attributes:
        dimension: The fixed number of components stored in the column.
    """

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        """Initialise the type.

        Args:
            dimension: Number of components in vectors stored in this column.

        Raises:
            ValueError: If ``dimension`` is not a positive integer.
        """
        if dimension <= 0:
            raise ValueError("Vector dimension must be a positive integer.")
        self.dimension = dimension

    def get_col_spec(self, **kw: object) -> str:
        """Render the column type for DDL.

        Args:
            **kw: Dialect keyword arguments, unused.

        Returns:
            The MariaDB column specification, for example ``VECTOR(384)``.
        """
        del kw
        return f"VECTOR({self.dimension})"

    def bind_expression(self, bindvalue: ColumnElement[Any]) -> ColumnElement[Any]:
        """Wrap the bound parameter so MariaDB parses it as a vector.

        Args:
            bindvalue: The bound parameter carrying the serialised vector.

        Returns:
            A ``VEC_FromText(...)`` call around the bound parameter.
        """
        return func.VEC_FromText(bindvalue)

    def column_expression(self, column: ColumnElement[Any]) -> ColumnElement[Any]:
        """Wrap the column on read so MariaDB returns a parseable string.

        Args:
            column: The selected vector column.

        Returns:
            A ``VEC_ToText(...)`` call around the column.
        """
        return func.VEC_ToText(column)

    def bind_processor(
        self, dialect: Dialect
    ) -> Callable[[Sequence[float] | None], str | None]:
        """Return a callable serialising Python vectors for the database.

        Args:
            dialect: The active SQLAlchemy dialect, unused.

        Returns:
            A callable converting a sequence of floats to its text form.
        """
        del dialect
        dimension = self.dimension

        def process(value: Sequence[float] | None) -> str | None:
            """Serialise a vector to MariaDB's text representation.

            Args:
                value: A sequence of floats, or ``None``.

            Returns:
                The JSON array form of the vector, or ``None``.

            Raises:
                ValueError: If the vector's length does not match the column.
            """
            if value is None:
                return None
            components = [float(component) for component in value]
            if len(components) != dimension:
                raise ValueError(
                    f"Expected a {dimension}-dimensional vector, "
                    f"got {len(components)} dimensions."
                )
            return json.dumps(components)

        return process

    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> Callable[[str | bytes | None], tuple[float, ...] | None]:
        """Return a callable parsing database values into Python vectors.

        Args:
            dialect: The active SQLAlchemy dialect, unused.
            coltype: The DBAPI column type, unused.

        Returns:
            A callable converting MariaDB's text form to a tuple of floats.
        """
        del dialect, coltype

        def process(value: str | bytes | None) -> tuple[float, ...] | None:
            """Parse MariaDB's text representation of a vector.

            Args:
                value: The raw value returned by the driver.

            Returns:
                The vector as a tuple of floats, or ``None``.
            """
            if value is None:
                return None
            if isinstance(value, bytes | bytearray):
                value = value.decode("utf-8")
            return tuple(float(component) for component in json.loads(value))

        return process
