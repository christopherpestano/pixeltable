"""
Migration for primary key index enforcement (PR #1203).

Existing tables with is_pk=True columns but no primary_index_md need:
1. A PrimaryIndexMd generated and persisted
2. A physical partial unique index created in PostgreSQL

If the index creation fails due to duplicate live rows, the migration strips
is_pk from all columns and removes primary_index_md, preserving the data
without silent deduplication.
"""

import logging
from uuid import UUID

import psycopg.errors
import sqlalchemy as sql

from pixeltable.index.btree import BtreeIndex
from pixeltable.metadata import register_converter
from pixeltable.metadata.converters.util import convert_table_md
from pixeltable.metadata.schema import Table

_logger = logging.getLogger('pixeltable')

_MAX_VERSION = Table.MAX_VERSION  # 2^63 - 1


@register_converter(version=48)
def _(engine: sql.engine.Engine) -> None:
    convert_table_md(engine, table_md_updater=_add_primary_index_md, table_modifier=_create_pk_index)


def _add_primary_index_md(table_md: dict, table_id: UUID) -> None:
    """Add primary_index_md to tables that have is_pk columns but no primary_index_md."""
    # Skip if already has primary_index_md
    if table_md.get('primary_index_md') is not None:
        return

    # Skip views — only base tables get the physical index
    if table_md.get('view_md') is not None:
        return

    # Collect columns with is_pk=True
    pk_col_ids = [int(col_id) for col_id, col_md in table_md['column_md'].items() if col_md.get('is_pk', False)]

    if len(pk_col_ids) == 0:
        return

    # Generate PrimaryIndexMd, mirroring create_initial_md() logic
    next_idx_id = table_md['next_idx_id']
    table_md['primary_index_md'] = {
        'id': next_idx_id,
        'name': f'pk{table_id.hex}',
        'indexed_col_tbl_id': str(table_id),
        'indexed_col_ids': pk_col_ids,
    }
    table_md['next_idx_id'] = next_idx_id + 1


def _create_pk_index(conn: sql.Connection, tbl_id: UUID, orig_table_md: dict, updated_table_md: dict) -> None:
    """Attempt to create the physical partial unique index. On duplicate rows, strip PK."""
    primary_index_md = updated_table_md.get('primary_index_md')

    # Nothing to do if we didn't add a primary_index_md in the updater
    if primary_index_md is None:
        return
    # Nothing to do if the original already had one (index already exists)
    if orig_table_md.get('primary_index_md') is not None:
        return

    tbl_name = updated_table_md['name']
    store_name = f'tbl_{tbl_id.hex}'
    idx_name = f'pk_idx_{tbl_id.hex}'
    pk_col_ids = primary_index_md['indexed_col_ids']

    # Build the index column expressions, applying left(col, 256) for string columns
    idx_col_exprs: list[str] = []
    for col_id in pk_col_ids:
        col_md = updated_table_md['column_md'][str(col_id)]
        col_store_name = f'col_{col_id}'
        if col_md['col_type'].get('_classname') == 'StringType':
            idx_col_exprs.append(f'left({col_store_name}, {BtreeIndex.MAX_STRING_LEN})')
        else:
            idx_col_exprs.append(col_store_name)

    cols_sql = ', '.join(idx_col_exprs)
    create_idx_sql = (
        f'CREATE UNIQUE INDEX {idx_name} ON {store_name} USING btree ({cols_sql}) WHERE v_max = {_MAX_VERSION}'
    )

    # Use a savepoint so a failure doesn't abort the outer transaction
    try:
        conn.execute(sql.text('SAVEPOINT pk_migration'))
        conn.execute(sql.text(create_idx_sql))
        conn.execute(sql.text('RELEASE SAVEPOINT pk_migration'))
        _logger.info(f'Migrated primary key index for table {tbl_name}')
    except sql.exc.IntegrityError as e:
        conn.execute(sql.text('ROLLBACK TO SAVEPOINT pk_migration'))
        if isinstance(e.orig, psycopg.errors.UniqueViolation):
            _logger.warning(
                f'Table {tbl_name} had primary_key columns with duplicate rows. '
                f'Primary key constraint removed. To re-enable, deduplicate rows and '
                f'recreate with primary_key=...'
            )
            # Strip is_pk from all columns
            for col_md in updated_table_md['column_md'].values():
                col_md['is_pk'] = False
            # Remove primary_index_md
            updated_table_md['primary_index_md'] = None
            # Persist the stripped metadata
            conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=updated_table_md))
        else:
            raise
