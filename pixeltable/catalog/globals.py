"""
Globals - Shared constants, enums, and utility functions for the catalog module.

Contains:
- QColumnId: Qualified column identifier (table UUID + column int id)
- MediaValidation: Enum controlling when media files are validated (on read vs. write)
- IfExistsParam / IfNotExistsParam: Enums for idempotent schema operations
- is_valid_identifier: Validates Python-identifier-style names for tables/columns
- is_system_column_name: Checks if a name conflicts with built-in Table/View methods
- Reserved column names (_POS_COLUMN_NAME, _ROWID_COLUMN_NAME)
"""

from __future__ import annotations

import enum
import itertools
import logging
from dataclasses import dataclass
from uuid import UUID

import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')

# name of the position column in a component view
_POS_COLUMN_NAME = 'pos'
_ROWID_COLUMN_NAME = '_rowid'

# Set of symbols that are predefined in the `InsertableTable` class (and are therefore not allowed as column names).
# This will be populated lazily to avoid circular imports.
_PREDEF_SYMBOLS: set[str] | None = None


@dataclass(frozen=True)
class QColumnId:
    """Qualified column id"""

    tbl_id: UUID
    col_id: int


class MediaValidation(enum.Enum):
    """Controls when media file integrity is checked.

    ON_READ: Validate media when data is queried (lazy, avoids insert-time overhead).
    ON_WRITE: Validate media at insert time (eager, catches errors early).
    """

    ON_READ = 0
    ON_WRITE = 1

    @classmethod
    def validated(cls, name: str, error_prefix: str) -> MediaValidation:
        try:
            return cls[name.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{error_prefix} must be one of: [{val_strs}]') from None


class IfExistsParam(enum.Enum):
    """Controls behavior when a schema object already exists at the target path.

    ERROR: Raise an exception (default).
    IGNORE: Silently do nothing and return the existing object.
    REPLACE: Drop the existing object and create a new one (if no dependents).
    REPLACE_FORCE: Like REPLACE but also drops dependents.
    """

    ERROR = 0
    IGNORE = 1
    REPLACE = 2
    REPLACE_FORCE = 3

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{param_name} must be one of: [{val_strs}]') from None


class IfNotExistsParam(enum.Enum):
    """Controls behavior when the target schema object does not exist.

    ERROR: Raise an exception (default).
    IGNORE: Silently do nothing.
    """

    ERROR = 0
    IGNORE = 1

    @classmethod
    def validated(cls, param_val: str, param_name: str) -> IfNotExistsParam:
        try:
            return cls[param_val.upper()]
        except KeyError:
            val_strs = ', '.join(f'{s.lower()!r}' for s in cls.__members__)
            raise excs.Error(f'{param_name} must be one of: [{val_strs}]') from None


def is_valid_identifier(name: str, *, allow_system_identifiers: bool = False, allow_hyphens: bool = False) -> bool:
    # If allow_hyphens=True, we allow hyphens to appear in the name, but we still do not permit a name to start with
    # one (even if allow_system_identifiers=True)
    adj_name = name.replace('-', '_') if allow_hyphens else name
    return (
        adj_name.isidentifier() and not name.startswith('-') and (allow_system_identifiers or not name.startswith('_'))
    )


def is_system_column_name(name: str) -> bool:
    """Returns True if the name conflicts with built-in Table/View attributes.

    These names are reserved because they would shadow Python methods (e.g., 'select',
    'insert', 'columns') when accessed via t.name attribute syntax.
    """
    from pixeltable.catalog import InsertableTable, View

    global _PREDEF_SYMBOLS  # noqa: PLW0603
    if _PREDEF_SYMBOLS is None:
        _PREDEF_SYMBOLS = set(itertools.chain(dir(InsertableTable), dir(View)))
    return name in _PREDEF_SYMBOLS
