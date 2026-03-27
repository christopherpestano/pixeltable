"""Tests for the primary key index migration (convert_48)."""

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
        # Downgrade schema version so the converter runs on next init
        conn.execute(SystemInfo.__table__.update().values(md={'schema_version': _PRE_MIGRATION_VERSION}))


def _pg_index_exists(tbl_id_hex: str) -> bool:
    """Check if the physical PK index exists in PostgreSQL."""
    idx_name = f'pk_idx_{tbl_id_hex}'
    with get_runtime().begin_xact() as conn:
        result = conn.execute(sql.text(f"SELECT COUNT(*) FROM pg_indexes WHERE indexname = '{idx_name}'")).scalar()
        return result > 0


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
        """Table with PK and no duplicates: migration adds the index successfully."""
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

        # Reload catalog — this triggers Env._init_env() which runs upgrade_md()
        reload_catalog()

        # Verify the migration re-created the index
        assert _pg_index_exists(tbl_id_hex)

        # Verify primary_index_md was restored in stored metadata
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is not None
        assert stored_md['primary_index_md']['indexed_col_ids'] == [0]

        # Verify PK enforcement works
        t = pxt.get_table('test_pk_migrate')
        with pytest.raises(pxt.Error, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'dupe'}])
        assert t.count() == 2

    def test_migration_with_duplicates(self, uses_db: None) -> None:
        """Table with PK but duplicate rows: migration strips is_pk and preserves data."""
        t = pxt.create_table('test_pk_dupes', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        validate_update_status(t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}]), expected_rows=2)

        tbl_id_hex = t._tbl_version.get().id.hex
        store_name = f'tbl_{tbl_id_hex}'

        # Simulate pre-migration state
        _simulate_pre_migration_state(tbl_id_hex)

        # Insert a duplicate row directly into the store table (bypassing PK enforcement)
        engine = Env.get().engine
        with engine.begin() as conn:
            # Find the max rowid to assign a new one
            max_rowid = conn.execute(sql.text(f'SELECT MAX(rowid) FROM {store_name}')).scalar()
            # Get the current version from table metadata
            stored_md = _get_stored_table_md(tbl_id_hex)
            current_version = stored_md['current_version']
            # Insert a duplicate: same id=1 but different rowid, as a live row
            conn.execute(
                sql.text(
                    f'INSERT INTO {store_name} (rowid, v_min, v_max, col_0, col_1) '
                    f"VALUES ({max_rowid + 1}, {current_version}, {Table.MAX_VERSION}, 1, 'duplicate_alice')"
                )
            )

        # Reload catalog — triggers migration
        reload_catalog()

        # The migration should have stripped is_pk because of the duplicate
        stored_md = _get_stored_table_md(tbl_id_hex)
        assert stored_md['primary_index_md'] is None
        for col_md in stored_md['column_md'].values():
            assert col_md['is_pk'] is False

        # The physical PK index should NOT exist
        assert not _pg_index_exists(tbl_id_hex)

        # All rows (including the duplicate) should be preserved
        t = pxt.get_table('test_pk_dupes')
        assert t.count() == 3

        # Duplicates are allowed now since PK was stripped
        validate_update_status(t.insert([{'id': 1, 'name': 'yet_another'}]), expected_rows=1)
        assert t.count() == 4

    def test_migration_composite_pk(self, uses_db: None) -> None:
        """Composite PK migration: happy path with multiple PK columns."""
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

        reload_catalog()

        # Index was recreated
        assert _pg_index_exists(tbl_id_hex)

        # PK enforcement works after migration
        t = pxt.get_table('test_composite_migrate')
        with pytest.raises(pxt.Error, match='Duplicate primary key'):
            t.insert([{'a': 1, 'b': 'x', 'val': 30}])
        assert t.count() == 2

    def test_migration_idempotent(self, uses_db: None) -> None:
        """Migration is idempotent: running on a table that already has primary_index_md is a no-op."""
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
