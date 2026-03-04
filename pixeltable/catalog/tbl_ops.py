"""
Table Operations - Crash-recoverable operation log for schema/data mutations.

This module implements the "pending table ops" mechanism that provides crash recovery
for multi-step table mutations (create table, add column, create view, drop table, etc.).

How it works:
1. A mutation writes an ordered list of TableOps to the PendingTableOp table in a single
   transaction, along with setting the table state to ROLLFORWARD.
2. The Catalog then executes each op in sequence, marking them COMPLETED.
3. If all ops succeed, the ops are deleted and the table state returns to LIVE.
4. If an op fails and the statement is abortable, the table state switches to ROLLBACK,
   and completed ops are undone in reverse order.

Each TableOp subclass declares:
- needs_tv: Whether it requires a TableVersion instance to execute
- needs_xact: Whether it must run inside a transaction (vs. outside for DDL)
- exec(): Forward execution logic
- undo(): Reverse execution logic for rollback

This design ensures that even if Pixeltable crashes mid-operation, the next access
to the table will detect the pending ops and finalize them.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import sqlalchemy as sql

import pixeltable.metadata.schema as schema
from pixeltable.runtime import get_runtime

from .update_status import UpdateStatus

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion

_logger = logging.getLogger('pixeltable')


class OpStatus(Enum):
    """Lifecycle state of an individual table operation.

    PENDING: Not yet executed (initial state).
    COMPLETED: Successfully executed during rollforward.
    ABORTED: Successfully undone during rollback.
    """

    PENDING = 0
    COMPLETED = 1
    ABORTED = 2


@dataclasses.dataclass
class TableOp:
    """TableOp describes an individual operation that needs to be performed on the table.

    If needs_xact is True, the TableOp is executed, and its state is updated as part of a single store transaction.
    Otherwise, the op is executed outside of the store transaction. Such operations (including undo) must be idempotent
    and safe to execute concurrently, because multiple processes may attempt to make progress on the same TableOp
    simultaneously.
    """

    needs_tv: ClassVar[bool]  # if False, exec/undo can be called with tv=None
    needs_xact: ClassVar[bool]  # whether this op must run as part of a transaction

    tbl_id: str  # uuid.UUID
    op_sn: int  # sequence number within the update operation; [0, num_ops)
    num_ops: int  # total number of ops forming the update operation
    status: OpStatus

    def to_dict(self) -> dict:
        result = dataclasses.asdict(self, dict_factory=schema.md_dict_factory)
        result['_classname'] = self.__class__.__name__
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TableOp:
        classname = data.pop('_classname')
        # needs_xact used to be a member variable. Remove it from the dict for backward compatibility.
        # TODO: delete this line, and the assert the follows, in ~ May 2026 or later. The chance of anyone still having
        # needs_xact in their pending table ops at that point will be extremely low.
        needs_xact_legacy = data.pop('needs_xact', None)
        op_class = getattr(sys.modules[__name__], classname)
        op = schema.md_from_dict(op_class, data)
        if needs_xact_legacy is not None:
            assert op.needs_xact == needs_xact_legacy
        return op

    def exec(self, tv: TableVersion | None) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.exec()')

    def undo(self, tv: TableVersion | None) -> None:
        raise NotImplementedError(f'{self.__class__.__name__}.undo()')


@dataclasses.dataclass
class CreateStoreTableOp(TableOp):
    """Creates the physical PostgreSQL store table for a new table/view.

    Runs outside a transaction because DDL (CREATE TABLE) auto-commits in PostgreSQL.
    Undo drops the store table.
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        with get_runtime().begin_xact():
            tv.store_tbl.create()

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        with get_runtime().begin_xact():
            tv.store_tbl.drop()


@dataclasses.dataclass
class CreateStoreIdxsOp(TableOp):
    """Creates physical database indices (e.g., B-tree) on the store table.

    Each index is created in its own transaction since DDL is auto-commit.
    Undo drops each index.
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    idx_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for idx_id in self.idx_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.create_index(idx_id)

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for idx_id in self.idx_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.drop_index(idx_id)


@dataclasses.dataclass
class LoadViewOp(TableOp):
    """Populates a newly created view by executing the view load plan.

    Reads matching rows from the base table and inserts them into the view's store table.
    Runs inside a transaction because it writes data rows.
    Undo deletes any media files and clears the file cache.
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = True

    view_path: dict[str, Any]  # needed to create the view load plan

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.catalog.table_version_path import TableVersionPath
        from pixeltable.plan import Planner

        assert get_runtime().in_xact
        view_path = TableVersionPath.from_dict(self.view_path)
        plan, _ = Planner.create_view_load_plan(view_path)
        with get_runtime().report_progress():
            plan.ctx.title = tv.display_str()
            _, row_counts = tv.store_tbl.insert_rows(plan, v_min=tv.version)
        status = UpdateStatus(row_count_stats=row_counts)
        get_runtime().catalog.store_update_status(tv.id, tv.version, status)
        _logger.debug(f'Loaded view {tv.name} with {row_counts.num_rows} rows')

    def undo(self, tv: TableVersion | None) -> None:
        from pixeltable.utils.filecache import FileCache

        # clear out any media files
        tv.delete_media()
        FileCache.get().clear(tbl_id=tv.id)


@dataclasses.dataclass
class CreateTableMdOp(TableOp):
    """Undo-only log record"""

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_tbl_md(uuid.UUID(self.tbl_id))


@dataclasses.dataclass
class DeleteTableMdOp(TableOp):
    """Deletes all metadata records for a table from the catalog store.

    This is the final step of table drop. It cannot be undone (irreversible).
    """

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_tbl_md(uuid.UUID(self.tbl_id))

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()


@dataclasses.dataclass
class CreateTableVersionOp(TableOp):
    """Undo-only log record for version creation.

    exec() is a no-op because the version metadata was already written in the initial transaction.
    undo() removes the version metadata to revert the schema change.
    """

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = True

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        assert get_runtime().in_xact
        get_runtime().catalog.delete_current_tbl_version_md(uuid.UUID(self.tbl_id))


@dataclasses.dataclass
class CreateColumnMdOp(TableOp):
    """Undo-only log record for column creation.

    exec() is a no-op because column metadata was written in the initial transaction.
    undo() removes the column metadata records to revert the add-column operation.
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = True

    column_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        pass

    def undo(self, tv: TableVersion | None) -> None:
        # TODO this is completely broken, but the fix requires a separate change and more thought. Leaving as is for now
        # because this change is meant to be mostly a refactoring (and a minor change in behavior, but elsewhere)
        # 1. major: write_tbl_md cannot be called while there are pending ops (and we are inside one of them).
        # 2. minor: [] is not an acceptable value for pending_ops
        # 3. minor: TableVersion internals access. Once we figure out how to fix 1, this one should go away as well.
        assert get_runtime().in_xact
        for col_id in self.column_ids:
            del tv._tbl_md.column_md[col_id]
        get_runtime().catalog.write_tbl_md(tv.id, None, tv._tbl_md, None, None, [])


@dataclasses.dataclass
class CreateStoreColumnsOp(TableOp):
    """Adds physical columns to the store table via ALTER TABLE.

    Runs outside a transaction because DDL is auto-commit.
    Each column is added individually with if_not_exists for idempotency.
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    column_ids: list[int]

    def exec(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for col_id in self.column_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.add_column(tv.cols_by_id[col_id], if_not_exists=True)

    def undo(self, tv: TableVersion | None) -> None:
        assert not get_runtime().in_xact
        for col_id in self.column_ids:
            with get_runtime().begin_xact():
                tv.store_tbl.drop_column(tv.cols_by_id[col_id], if_exists=True)


@dataclasses.dataclass
class DeleteTableMediaFilesOp(TableOp):
    """Deletes all media files (images, audio, etc.) stored for a table.

    This runs outside a transaction and clears both object store files and the local file cache.
    Cannot be undone (irreversible).
    """

    needs_tv: ClassVar[bool] = True
    needs_xact: ClassVar[bool] = False

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.utils.filecache import FileCache

        tv.delete_media()
        FileCache.get().clear(tbl_id=tv.id)

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()


@dataclasses.dataclass
class DropStoreTableOp(TableOp):
    """Drops the physical PostgreSQL store table via DROP TABLE IF EXISTS.

    Runs outside a transaction because DDL is auto-commit.
    Cannot be undone (irreversible).
    """

    needs_tv: ClassVar[bool] = False
    needs_xact: ClassVar[bool] = False

    is_view: bool

    def exec(self, tv: TableVersion | None) -> None:
        from pixeltable.store import StoreBase

        assert not get_runtime().in_xact
        with get_runtime().begin_xact() as conn:
            drop_stmt = f'DROP TABLE IF EXISTS {StoreBase.storage_name(uuid.UUID(self.tbl_id), self.is_view)}'
            conn.execute(sql.text(drop_stmt))

    def undo(self, tv: TableVersion | None) -> None:
        raise AssertionError()
