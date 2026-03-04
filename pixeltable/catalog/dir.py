"""
Dir - Represents a directory (namespace) in the Pixeltable catalog hierarchy.

Directories form a tree structure that organizes tables and views into namespaces.
Each user has a root directory, and tables/views are addressed by dot-separated paths
(e.g., 'my_dir.my_table').

Directories are SchemaObjects with:
- A UUID identifier
- A parent directory (except the root, which has parent_id=None)
- A name (empty string for the root directory)

Directory operations (create, move, drop) are coordinated by the Catalog class
with appropriate locking to prevent concurrent modifications.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime

from .schema_object import SchemaObject

_logger = logging.getLogger('pixeltable')


class Dir(SchemaObject):
    """A namespace directory in the Pixeltable catalog hierarchy.

    Directories organize tables, views, and subdirectories into a tree structure.
    Each directory is persisted as a row in the schema.Dir table in PostgreSQL.
    """

    def __init__(self, id: UUID, parent_id: UUID, name: str):
        super().__init__(id, name, parent_id)

    @classmethod
    def _create(cls, parent_id: UUID, name: str) -> Dir:
        """Create a new directory in the catalog store and return a Dir instance."""
        session = get_runtime().session
        user = Env.get().user
        assert session is not None
        dir_md = schema.DirMd(name=name, user=user, additional_md={})
        dir_record = schema.Dir(parent_id=parent_id, md=dataclasses.asdict(dir_md))
        session.add(dir_record)
        session.flush()
        # print(f'{datetime.datetime.now()} create dir {dir_record}')
        assert dir_record.id is not None
        assert isinstance(dir_record.id, UUID)
        dir = cls(dir_record.id, parent_id, name)
        return dir

    def _display_name(self) -> str:
        return 'directory'

    def _path(self) -> str:
        """Returns the path to this schema object."""
        if self._dir_id is None:
            # we're the root dir
            return ''
        return super()._path()

    def _move(self, new_name: str, new_dir_id: UUID) -> None:
        # print(
        #     f'{datetime.datetime.now()} move dir name={self._name} parent={self._dir_id} '
        #     f'new_name={new_name} new_dir_id={new_dir_id}'
        # )
        super()._move(new_name, new_dir_id)
        stmt = sql.text(
            (
                f'UPDATE {schema.Dir.__table__} '
                f'SET {schema.Dir.parent_id.name} = :new_dir_id, '
                f"    {schema.Dir.md.name} = jsonb_set({schema.Dir.md.name}, '{{name}}', (:new_name)::jsonb) "
                f'WHERE {schema.Dir.id.name} = :id'
            )
        )
        get_runtime().conn.execute(stmt, {'new_dir_id': new_dir_id, 'new_name': json.dumps(new_name), 'id': self._id})
