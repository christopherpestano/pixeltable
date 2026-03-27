"""Tests for the primary key index migration (convert_48).

These tests validate the migration against a real PostgreSQL instance by:
- Setting up pre-migration state (dropping indexes, clearing metadata, downgrading schema version)
- Triggering the migration via Env._init_env() + reload_catalog()
- Querying pg_indexes to verify physical index existence and definition
- Verifying PK enforcement and data integrity end-to-end
"""

import logging

import pytest
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.metadata.schema import SystemInfo, Table
from pixeltable.runtime import get_runtime

from .utils import reload_catalog, validate_update_status

_logger = logging.getLogger('pixeltable')

# The converter upgrades from version 48 to 49
_PRE_MIGRATION_VERSION = 48


def _simulate_pre_migration_state(tbl_id_hex: str) -> None:
    """Simulate a table that was created before the PK index migration.

    Removes the physical unique index, clears primary_index_md from stored metadata,
    and downgrades the schema version so that upgrade_md() will re-run the converter.
    """
    engine = Env.get().engine

    # Drop the physical PK index
    idx_name = f'pk_idx_{tbl_id_hex}'
    with engine.begin() as conn:
        conn.execute(sql.text(f'DROP INDEX IF EXISTS {idx_name}'))

    # Remove primary_index_md from the stored table metadata and downgrade schema version
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md)):
            if row[0].hex == tbl_id_hex:
                table_md = row[1]
                table_md['primary_index_md'] = None
                conn.execute(sql.update(Table).where(Table.id == row[0]).values(md=table_md))
                break
        # Downgrade schema version so the converter runs on next init.
        # Use DELETE + INSERT because clean_db() may have truncated SystemInfo.
        conn.execute(SystemInfo.__table__.delete())
        conn.execute(SystemInfo.__table__.insert().values(dummy=0, md={'schema_version': _PRE_MIGRATION_VERSION}))


def _reload_with_migration() -> None:
    """Re-initialize Env (triggering upgrade_md) then reload catalog.

    Env._init_env() runs _upgrade_metadata() which detects the downgraded schema version
    and executes the converter. reload_catalog() then creates a fresh catalog from the
    updated metadata.
    """
    Env._init_env()
    reload_catalog()


def _pg_index_exists(tbl_id_hex: str) -> bool:
    """Check if the physical PK index exists in PostgreSQL via pg_indexes."""
    idx_name = f'pk_idx_{tbl_id_hex}'
    with get_runtime().begin_xact() as conn:
        result = conn.execute(sql.text(f"SELECT COUNT(*) FROM pg_indexes WHERE indexname = '{idx_name}'")).scalar()
        return result > 0


def _get_pg_index_def(tbl_id_hex: str) -> str | None:
    """Get the full CREATE INDEX definition from pg_indexes for the PK index."""
    idx_name = f'pk_idx_{tbl_id_hex}'
    with get_runtime().begin_xact() as conn:
        return conn.execute(sql.text(f"SELECT indexdef FROM pg_indexes WHERE indexname = '{idx_name}'")).scalar()


def _get_stored_table_md(tbl_id_hex: str) -> dict:
    """Read the stored TableMd dict for a given table id."""
    engine = Env.get().engine
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md)):
            if row[0].hex == tbl_id_hex:
                return row[1]
    raise ValueError(f'Table {tbl_id_hex} not found')


class TestPkMigration:
    def test_migration_happy_path(self, uses_db: None) -> None:
        """Single-column PK, no duplicates: migration adds the index and enforces PK."""
        t = pxt.create_table('test_pk_migrate', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]), expected_rows=2)

        tbl_id_hex = t._tbl_version.get().id.hex

        # Verify index exists now
        assert _pg_index_exists(tbl_id_hex)

        # Simulate pre-migration state: remove the index and primary_index_md
        _simulate_pre_migration_state(tbl_id_hex)
        assert not _pg_index_exists(tbl_id_hex)

        # Verify metadata was cleared
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is None

        # Re-init env (triggers migration) and reload catalog
        _reload_with_migration()

        # Verify the physical index was re-created in PostgreSQL
        assert _pg_index_exists(tbl_id_hex)
        idx_def = _get_pg_index_def(tbl_id_hex)
        assert idx_def is not None
        assert 'UNIQUE' in idx_def.upper()
        assert 'col_0' in idx_def

        # Verify primary_index_md was restored in stored metadata
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is not None
        assert stored_md['primary_index_md']['indexed_col_ids'] == [0]

        # Verify PK enforcement works (duplicate insert fails)
        t = pxt.get_table('test_pk_migrate')
        with pytest.raises(pxt.Error, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
        assert t.count() == 2

        # Verify normal inserts still work after migration
        validate_update_status(t.insert([{'id': 3, 'name': 'charlie'}]), expected_rows=1)
        assert t.count() == 3

    def test_migration_with_duplicates(self, uses_db: None) -> None:
        """Single-column PK with duplicate rows: migration strips is_pk and preserves all data."""
        t = pxt.create_table('test_pk_dupes', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]), expected_rows=2)

        tbl_id_hex = t._tbl_version.get().id.hex
        store_name = f'tbl_{tbl_id_hex}'

        # Simulate pre-migration state
        _simulate_pre_migration_state(tbl_id_hex)

        # Insert a duplicate row directly via raw SQL (bypassing Pixeltable)
        engine = Env.get().engine
        with engine.begin() as conn:
            max_rowid = conn.execute(sql.text(f'SELECT MAX(rowid) FROM {store_name}')).scalar()
            stored_md = _get_stored_table_md(tbl_id_hex)
            current_version = stored_md['current_version']
            conn.execute(
                sql.text(
                    f'INSERT INTO {store_name} (rowid, v_min, v_max, col_0, col_1) '
                    f"VALUES ({max_rowid + 1}, {current_version}, {Table.MAX_VERSION}, 1, 'duplicate_alice')"
                )
            )

        # Re-init env (triggers migration) and reload catalog
        _reload_with_migration()

        # Verify NO physical PK index exists in PostgreSQL
        assert not _pg_index_exists(tbl_id_hex)
        assert _get_pg_index_def(tbl_id_hex) is None

        # Verify metadata: is_pk stripped on all columns, primary_index_md removed
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is None
        for col_md in stored_md['column_md'].values():
            assert col_md['is_pk'] is False

        # All rows (including the duplicate) must be preserved
        t = pxt.get_table('test_pk_dupes')
        assert t.count() == 3

        # Inserts work without PK enforcement
        validate_update_status(t.insert([{'id': 1, 'name': 'yet_another'}]), expected_rows=1)
        assert t.count() == 4

    def test_migration_composite_pk_with_string(self, uses_db: None) -> None:
        """Composite PK (int + string): migration creates index with left(col, 256) truncation."""
        t = pxt.create_table(
            'test_composite_migrate',
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String], 'val': pxt.Int},
            primary_key=['a', 'b'],
        )
        validate_update_status(
            t.insert([{'a': 1, 'b': 'x', 'val': 10}, {'a': 1, 'b': 'y', 'val': 20}]), expected_rows=2
        )
        tbl_id_hex = t._tbl_version.get().id.hex

        _simulate_pre_migration_state(tbl_id_hex)
        assert not _pg_index_exists(tbl_id_hex)

        _reload_with_migration()

        # Physical index was recreated
        assert _pg_index_exists(tbl_id_hex)

        # Verify the index definition: int column is plain, string column uses left(col, 256)
        idx_def = _get_pg_index_def(tbl_id_hex)
        assert idx_def is not None
        assert 'col_0' in idx_def  # int column (a)
        assert '"left"' in idx_def and 'col_1' in idx_def  # string column (b) uses left() truncation
        assert '256' in idx_def  # truncation length matches BtreeIndex.MAX_STRING_LEN

        # Verify metadata
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is not None
        assert sorted(stored_md['primary_index_md']['indexed_col_ids']) == [0, 1]

        # PK enforcement works after migration
        t = pxt.get_table('test_composite_migrate')
        with pytest.raises(pxt.Error, match='Duplicate primary key'):
            t.insert([{'a': 1, 'b': 'x', 'val': 30}])
        assert t.count() == 2

        # Normal inserts still work
        validate_update_status(t.insert([{'a': 2, 'b': 'z', 'val': 40}]), expected_rows=1)
        assert t.count() == 3

    def test_migration_composite_pk_with_string_duplicates(self, uses_db: None) -> None:
        """Composite PK (int + string) with duplicates: migration strips PK, preserves all rows."""
        t = pxt.create_table(
            'test_composite_dupes',
            {'a': pxt.Required[pxt.Int], 'b': pxt.Required[pxt.String], 'val': pxt.Int},
            primary_key=['a', 'b'],
        )
        validate_update_status(
            t.insert([{'a': 1, 'b': 'x', 'val': 10}, {'a': 1, 'b': 'y', 'val': 20}]), expected_rows=2
        )
        tbl_id_hex = t._tbl_version.get().id.hex
        store_name = f'tbl_{tbl_id_hex}'

        _simulate_pre_migration_state(tbl_id_hex)

        # Insert a duplicate row via raw SQL (same composite key a=1, b='x')
        engine = Env.get().engine
        with engine.begin() as conn:
            max_rowid = conn.execute(sql.text(f'SELECT MAX(rowid) FROM {store_name}')).scalar()
            stored_md = _get_stored_table_md(tbl_id_hex)
            current_version = stored_md['current_version']
            conn.execute(
                sql.text(
                    f'INSERT INTO {store_name} (rowid, v_min, v_max, col_0, col_1, col_2) '
                    f"VALUES ({max_rowid + 1}, {current_version}, {Table.MAX_VERSION}, 1, 'x', 99)"
                )
            )

        _reload_with_migration()

        # No physical PK index in PostgreSQL
        assert not _pg_index_exists(tbl_id_hex)
        assert _get_pg_index_def(tbl_id_hex) is None

        # Metadata: PK stripped
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is None
        for col_md in stored_md['column_md'].values():
            assert col_md['is_pk'] is False

        # All rows preserved (including the duplicate)
        t = pxt.get_table('test_composite_dupes')
        assert t.count() == 3

        # Inserts work without PK enforcement
        validate_update_status(t.insert([{'a': 1, 'b': 'x', 'val': 999}]), expected_rows=1)
        assert t.count() == 4

    def test_migration_idempotent(self, uses_db: None) -> None:
        """Reload on a table that already has primary_index_md is a no-op."""
        t = pxt.create_table('test_pk_idempotent', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}]), expected_rows=1)

        tbl_id_hex = t._tbl_version.get().id.hex

        # Verify index and metadata exist
        assert _pg_index_exists(tbl_id_hex)
        stored_md = _get_stored_table_md(tbl_id_hex)
        original_primary_index_md = stored_md['primary_index_md']
        assert original_primary_index_md is not None

        # Reload — should be a no-op since primary_index_md already exists
        reload_catalog()

        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] == original_primary_index_md
        assert _pg_index_exists(tbl_id_hex)

        t = pxt.get_table('test_pk_idempotent')
        with pytest.raises(pxt.Error, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
