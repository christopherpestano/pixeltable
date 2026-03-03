"""
Global constants and enumerations for expression operators.

This module defines the operator enums used by expression nodes throughout the exprs package:
- ``ComparisonOperator``: relational operators (<, <=, ==, !=, >, >=)
- ``LogicalOperator``: boolean connectives (AND, OR, NOT)
- ``ArithmeticOperator``: numeric operations (+, -, *, /, %, //)
- ``StringOperator``: string operations (concatenation, repetition)

It also defines ``LiteralPythonTypes``, the union of Python types that can be wrapped
in a ``Literal`` expression, and the ``print_slice()`` utility for displaying slice objects.
"""
from __future__ import annotations

import datetime
import enum
import uuid

# Python types corresponding to our literal types
LiteralPythonTypes = str | int | float | bool | datetime.datetime | datetime.date | uuid.UUID


def print_slice(s: slice) -> str:
    """Format a slice object as a human-readable string (e.g., '1:3', '::2')."""
    start_str = f'{str(s.start) if s.start is not None else ""}'
    stop_str = f'{str(s.stop) if s.stop is not None else ""}'
    step_str = f'{str(s.step) if s.step is not None else ""}'
    return f'{start_str}:{stop_str}{":" if s.step is not None else ""}{step_str}'


class ComparisonOperator(enum.Enum):
    """Relational comparison operators used by ``Comparison`` expressions.

    Each variant maps to both a Python operator and a SQL operator.
    The ``reverse()`` method swaps operand order (e.g., LT <-> GT)
    to normalize search argument comparisons.
    """

    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5

    def __str__(self) -> str:
        if self == self.LT:
            return '<'
        if self == self.LE:
            return '<='
        if self == self.EQ:
            return '=='
        if self == self.NE:
            return '!='
        if self == self.GT:
            return '>'
        if self == self.GE:
            return '>='
        raise AssertionError()

    def reverse(self) -> ComparisonOperator:
        """Return the operator with swapped operand order.

        Used to normalize comparisons into (column, literal) order for index lookups.
        EQ and NE are symmetric and return themselves.
        """
        if self == self.LT:
            return self.GT
        if self == self.LE:
            return self.GE
        if self == self.GT:
            return self.LT
        if self == self.GE:
            return self.LE
        return self


class LogicalOperator(enum.Enum):
    """Boolean logical operators used by ``CompoundPredicate`` expressions.

    AND and OR are binary (variadic), NOT is unary.
    These map to Python's ``&``, ``|``, and ``~`` operators on Expr objects.
    """

    AND = 0
    OR = 1
    NOT = 2

    def __str__(self) -> str:
        if self == self.AND:
            return '&'
        if self == self.OR:
            return '|'
        if self == self.NOT:
            return '~'
        raise AssertionError()


class ArithmeticOperator(enum.Enum):
    """Numeric arithmetic operators used by ``ArithmeticExpr`` expressions.

    Each variant maps to both a Python operator and a SQL expression.
    DIV always produces float results. FLOORDIV uses SQL FLOOR to match
    Python's // semantics (round toward negative infinity).
    """

    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    FLOORDIV = 5

    def __str__(self) -> str:
        if self == self.ADD:
            return '+'
        if self == self.SUB:
            return '-'
        if self == self.MUL:
            return '*'
        if self == self.DIV:
            return '/'
        if self == self.MOD:
            return '%'
        if self == self.FLOORDIV:
            return '//'
        raise AssertionError()


class StringOperator(enum.Enum):
    """String operators used by ``StringOp`` expressions.

    CONCAT: string + string (SQL: ``left || right``)
    REPEAT: string * int (SQL: ``repeat(left, right)``)
    """

    CONCAT = 0
    REPEAT = 1

    def __str__(self) -> str:
        if self == self.CONCAT:
            return '+'
        if self == self.REPEAT:
            return '*'
        raise AssertionError()
