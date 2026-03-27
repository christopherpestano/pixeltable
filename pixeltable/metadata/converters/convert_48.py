import copy
import logging
from uuid import UUID

import sqlalchemy as sql

from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md
from pixeltable.metadata.schema import Table

_logger = logging.getLogger('pixeltable')

_MAX_VERSION = 9223372036854775807  # 2^63 - 1  (Table.MAX_VERSION)
_MAX_STRING_LEN = 256  # BtreeIndex.MAX_STRING_LEN


@register_converter(version=48)
def _(engine: sql.engine.Engine) -> None:
    """Backfill PrimaryIndexMd and physical unique index for pre-existing tables with is_pk columns."""
    convert_table_md(engine, table_modifier=_table_modifier)


def _table_modifier(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    # Skip views — only base tables can have primary keys
    if orig_table_md.get('view_md') is not None:
        return

    # Skip tables that already have a PrimaryIndexMd
    if orig_table_md.get('primary_index_md') is not None:
        return

    # Collect columns marked as PK
    pk_col_ids: list[int] = []
    pk_col_is_string: dict[int, bool] = {}
    for col_md in orig_table_md['column_md'].values():
        if col_md.get('is_pk', False):
            col_id = col_md['id']
            pk_col_ids.append(col_id)
            pk_col_is_string[col_id] = col_md.get('col_type', {}).get('_classname') == 'StringType'

    if not pk_col_ids:
        return  # No PK columns, nothing to do

    tbl_id_hex = tbl_id.hex
    store_name = f'tbl_{tbl_id_hex}'
    idx_name = f'pk_idx_{tbl_id_hex}'

    # Check if the physical store table exists
    row = conn.execute(
        sql.text("SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = :name"), {'name': store_name}
    ).fetchone()
    if row is None:
        _logger.warning(f'Store table {store_name} does not exist; skipping PK migration for table {tbl_id}')
        return

    # Idempotency: if the index already exists, just ensure metadata is in sync
    row = conn.execute(
        sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
        {'tbl': store_name, 'idx': idx_name},
    ).fetchone()
    if row is not None:
        _logger.info(f'Index {idx_name} already exists; persisting PrimaryIndexMd only')
        _persist_primary_index_md(conn, tbl_id, orig_table_md, pk_col_ids)
        return

    # Build the column expression list for the unique index
    idx_col_exprs: list[str] = []
    for col_id in pk_col_ids:
        col_name = f'col_{col_id}'
        if pk_col_is_string[col_id]:
            idx_col_exprs.append(f'left({col_name}, {_MAX_STRING_LEN})')
        else:
            idx_col_exprs.append(col_name)

    create_idx_sql = (
        f'CREATE UNIQUE INDEX {idx_name} ON {store_name} '
        f'USING btree ({", ".join(idx_col_exprs)}) '
        f'WHERE v_max = {_MAX_VERSION}'
    )

    # Use SAVEPOINT so a failure (e.g. duplicate rows) does not abort the outer transaction
    try:
        conn.execute(sql.text('SAVEPOINT pk_migration'))
        conn.execute(sql.text(create_idx_sql))
        conn.execute(sql.text('RELEASE SAVEPOINT pk_migration'))
        _logger.info(f'Created unique index {idx_name} on {store_name}')
        _persist_primary_index_md(conn, tbl_id, orig_table_md, pk_col_ids)
    except Exception as e:
        conn.execute(sql.text('ROLLBACK TO SAVEPOINT pk_migration'))
        _logger.warning(
            f'Cannot create unique index on {store_name} (likely duplicate rows): {e}. '
            f'Stripping PK designation from columns.'
        )
        _strip_pk(conn, tbl_id, orig_table_md)


def _persist_primary_index_md(conn: sql.Connection, tbl_id: UUID, table_md: dict, pk_col_ids: list[int]) -> None:
    """Create PrimaryIndexMd and persist it into the table metadata."""
    md = copy.deepcopy(table_md)
    next_idx_id = md.get('next_idx_id', 0)
    md['primary_index_md'] = {
        'id': next_idx_id,
        'name': f'pk{tbl_id.hex}',
        'indexed_col_tbl_id': str(tbl_id),
        'indexed_col_ids': pk_col_ids,
    }
    md['next_idx_id'] = next_idx_id + 1
    conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=md))
    _logger.info(f'Persisted PrimaryIndexMd for table {tbl_id}')


def _strip_pk(conn: sql.Connection, tbl_id: UUID, table_md: dict) -> None:
    """Remove is_pk from all columns and clear primary_index_md."""
    md = copy.deepcopy(table_md)
    for col_md in md['column_md'].values():
        col_md['is_pk'] = False
    md['primary_index_md'] = None
    conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=md))
    _logger.warning(f'Stripped PK designation from all columns in table {tbl_id}')
