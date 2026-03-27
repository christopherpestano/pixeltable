"""Integration tests for the v48 → v49 primary key migration converter."""

import logging

import pytest
import sqlalchemy as sql
from sqlalchemy import orm

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.metadata import VERSION
from pixeltable.metadata.schema import SystemInfo, Table

from .utils import reload_catalog

_logger = logging.getLogger('pixeltable')

_MAX_VERSION = 9223372036854775807  # Table.MAX_VERSION


def _downgrade_to_v48(engine: sql.engine.Engine) -> None:
    """Set the schema version back to 48 to trigger migration on next init."""
    with orm.Session(engine) as session:
        info = session.query(SystemInfo).one()
        md = dict(info.md)
        md['schema_version'] = 48
        info.md = md
        session.commit()


def _get_table_md(engine: sql.engine.Engine, table_name: str) -> tuple | None:
    """Return (tbl_id, table_md) for the given table name, or None if not found."""
    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md)):
            tbl_id = row[0]
            table_md = row[1]
            if table_md.get('name') == table_name:
                return tbl_id, table_md
    return None


def _drop_pk_index(engine: sql.engine.Engine, tbl_id) -> None:
    """Drop the physical pk_idx from the database."""
    idx_name = f'pk_idx_{tbl_id.hex}'
    with engine.begin() as conn:
        conn.execute(sql.text(f'DROP INDEX IF EXISTS {idx_name}'))


def _strip_primary_index_md(engine: sql.engine.Engine, tbl_id) -> None:
    """Remove primary_index_md from the table metadata (simulating pre-migration state)."""
    import copy

    with engine.begin() as conn:
        for row in conn.execute(sql.select(Table.id, Table.md).where(Table.id == tbl_id)):
            md = copy.deepcopy(row[1])
            md['primary_index_md'] = None
            conn.execute(sql.update(Table).where(Table.id == tbl_id).values(md=md))


def _index_exists(engine: sql.engine.Engine, tbl_id) -> bool:
    """Check if the pk_idx exists in pg_indexes."""
    idx_name = f'pk_idx_{tbl_id.hex}'
    store_name = f'tbl_{tbl_id.hex}'
    with engine.begin() as conn:
        row = conn.execute(
            sql.text('SELECT 1 FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
            {'tbl': store_name, 'idx': idx_name},
        ).fetchone()
        return row is not None


def _simulate_pre_migration(engine: sql.engine.Engine, tbl_id) -> None:
    """Simulate pre-migration state: drop index, remove metadata, downgrade version."""
    _drop_pk_index(engine, tbl_id)
    _strip_primary_index_md(engine, tbl_id)
    _downgrade_to_v48(engine)


class TestPkMigration:
    """Tests for the v48 → v49 primary key migration converter."""

    def test_happy_path_single_int_pk(self, init_env: None) -> None:
        """Table with single int PK, no duplicates: index should be created and PrimaryIndexMd persisted."""
        engine = Env.get().engine

        # Create table with PK via normal API
        t = pxt.create_table('test_pk_happy', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}])

        result = _get_table_md(engine, 'test_pk_happy')
        assert result is not None
        tbl_id, _ = result

        # Verify index exists before migration
        assert _index_exists(engine, tbl_id), 'PK index should exist after table creation'

        # Simulate pre-migration state
        _simulate_pre_migration(engine, tbl_id)

        # Verify index is gone
        assert not _index_exists(engine, tbl_id), 'PK index should be gone after simulation'

        # Verify primary_index_md is gone
        _, md = _get_table_md(engine, 'test_pk_happy')
        assert md.get('primary_index_md') is None, 'primary_index_md should be None before migration'

        # Trigger migration
        Env._init_env()
        reload_catalog()

        # Verify schema version upgraded
        with orm.Session(Env.get().engine) as session:
            info = session.query(SystemInfo).one()
            assert info.md['schema_version'] == VERSION

        # Verify index was recreated
        engine = Env.get().engine
        assert _index_exists(engine, tbl_id), 'PK index should be recreated by migration'

        # Verify PrimaryIndexMd was persisted
        _, md = _get_table_md(engine, 'test_pk_happy')
        assert md.get('primary_index_md') is not None, 'primary_index_md should be set after migration'
        assert len(md['primary_index_md']['indexed_col_ids']) == 1

        # Verify PK enforcement works: inserting a duplicate should fail
        t = pxt.get_table('test_pk_happy')
        with pytest.raises(pxt.exceptions.Error, match='Duplicate primary key'):
            t.insert([{'id': 1, 'name': 'duplicate'}])

        pxt.drop_table('test_pk_happy', force=True)

    def test_duplicate_rows_strip_pk(self, init_env: None) -> None:
        """Table with duplicates: PK should be stripped, all rows preserved."""
        engine = Env.get().engine

        # Create table with PK
        t = pxt.create_table('test_pk_dupes', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}])

        result = _get_table_md(engine, 'test_pk_dupes')
        assert result is not None
        tbl_id, _ = result

        # Simulate pre-migration state
        _simulate_pre_migration(engine, tbl_id)

        # Insert a duplicate row via raw SQL (bypassing the now-dropped index)
        store_name = f'tbl_{tbl_id.hex}'
        with engine.begin() as conn:
            # Find the col_id for 'id' column
            _, md = _get_table_md(engine, 'test_pk_dupes')
            id_col_id = None
            name_col_id = None
            for col_md in md['column_md'].values():
                if col_md.get('is_pk', False):
                    id_col_id = col_md['id']
                elif col_md.get('col_type', {}).get('_classname') == 'StringType':
                    name_col_id = col_md['id']

            assert id_col_id is not None
            assert name_col_id is not None

            # Insert duplicate row with same id=1 via raw SQL
            conn.execute(
                sql.text(
                    f'INSERT INTO {store_name} (rowid, v_min, v_max, col_{id_col_id}, col_{name_col_id}) '
                    f'VALUES (999, 0, {_MAX_VERSION}, 1, :name)'
                ),
                {'name': 'duplicate_alice'},
            )

        # Trigger migration
        Env._init_env()
        reload_catalog()

        # Verify index was NOT created (because of duplicates)
        engine = Env.get().engine
        assert not _index_exists(engine, tbl_id), 'PK index should NOT exist when duplicates present'

        # Verify PK was stripped from metadata
        _, md = _get_table_md(engine, 'test_pk_dupes')
        assert md.get('primary_index_md') is None, 'primary_index_md should be None when dupes found'
        for col_md in md['column_md'].values():
            assert col_md.get('is_pk', False) is False, 'All columns should have is_pk=False'

        # Verify all rows are preserved (original 2 + duplicate = 3 live rows)
        with engine.begin() as conn:
            count = conn.execute(sql.text(f'SELECT COUNT(*) FROM {store_name} WHERE v_max = {_MAX_VERSION}')).scalar()
            assert count == 3, f'Expected 3 live rows, got {count}'

        pxt.drop_table('test_pk_dupes', force=True)

    def test_composite_pk_with_string(self, init_env: None) -> None:
        """Composite PK with int + string column: index should use left(col, 256) for strings."""
        engine = Env.get().engine

        # Create table with composite PK including a string column
        t = pxt.create_table(
            'test_pk_composite',
            {'region': pxt.Required[pxt.String], 'user_id': pxt.Required[pxt.Int], 'data': pxt.String},
            primary_key=['region', 'user_id'],
        )
        t.insert([{'region': 'us', 'user_id': 1, 'data': 'hello'}, {'region': 'eu', 'user_id': 1, 'data': 'world'}])

        result = _get_table_md(engine, 'test_pk_composite')
        assert result is not None
        tbl_id, _ = result

        # Simulate pre-migration state
        _simulate_pre_migration(engine, tbl_id)

        # Trigger migration
        Env._init_env()
        reload_catalog()

        # Verify index was created
        engine = Env.get().engine
        assert _index_exists(engine, tbl_id), 'Composite PK index should be created'

        # Verify PrimaryIndexMd has both columns
        _, md = _get_table_md(engine, 'test_pk_composite')
        pim = md.get('primary_index_md')
        assert pim is not None
        assert len(pim['indexed_col_ids']) == 2

        # Verify the index definition uses left() for the string column
        idx_name = f'pk_idx_{tbl_id.hex}'
        store_name = f'tbl_{tbl_id.hex}'
        with engine.begin() as conn:
            row = conn.execute(
                sql.text('SELECT indexdef FROM pg_indexes WHERE tablename = :tbl AND indexname = :idx'),
                {'tbl': store_name, 'idx': idx_name},
            ).fetchone()
            assert row is not None
            indexdef = row[0]
            # PostgreSQL may quote left as "left" in the index definition
            assert 'left' in indexdef.lower(), f'String PK column should use left() truncation: {indexdef}'

        pxt.drop_table('test_pk_composite', force=True)

    def test_table_without_pk_unchanged(self, init_env: None) -> None:
        """Table without PK columns should not be affected by migration."""
        engine = Env.get().engine

        # Create table without PK
        t = pxt.create_table('test_no_pk', {'id': pxt.Int, 'name': pxt.String})
        t.insert([{'id': 1, 'name': 'alice'}])

        result = _get_table_md(engine, 'test_no_pk')
        assert result is not None
        tbl_id, _orig_md = result

        # Downgrade version
        _downgrade_to_v48(engine)

        # Trigger migration
        Env._init_env()
        reload_catalog()

        # Verify table is unmodified
        engine = Env.get().engine
        _, md = _get_table_md(engine, 'test_no_pk')
        assert md.get('primary_index_md') is None
        assert not _index_exists(engine, tbl_id)

        pxt.drop_table('test_no_pk', force=True)

    def test_idempotent_migration(self, init_env: None) -> None:
        """Running migration twice should not cause errors or duplicate indexes."""
        engine = Env.get().engine

        t = pxt.create_table('test_pk_idempotent', {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id')
        t.insert([{'id': 1, 'name': 'alice'}])

        result = _get_table_md(engine, 'test_pk_idempotent')
        assert result is not None
        tbl_id, _ = result

        # Simulate pre-migration state
        _simulate_pre_migration(engine, tbl_id)

        # Run migration twice
        Env._init_env()
        reload_catalog()

        # Downgrade again and run migration again
        engine = Env.get().engine
        _downgrade_to_v48(engine)
        Env._init_env()
        reload_catalog()

        # Verify everything is consistent
        engine = Env.get().engine
        assert _index_exists(engine, tbl_id)
        _, md = _get_table_md(engine, 'test_pk_idempotent')
        assert md.get('primary_index_md') is not None

        pxt.drop_table('test_pk_idempotent', force=True)

    def test_migration_multiple_tables(self, init_env: None) -> None:
        """Migration handles multiple tables: one with dupes (fails), one without (succeeds)."""
        engine = Env.get().engine

        # Create two tables with PK
        t1 = pxt.create_table('test_pk_multi_good', {'id': pxt.Required[pxt.Int], 'val': pxt.String}, primary_key='id')
        t1.insert([{'id': 1, 'val': 'a'}, {'id': 2, 'val': 'b'}])

        t2 = pxt.create_table('test_pk_multi_bad', {'id': pxt.Required[pxt.Int], 'val': pxt.String}, primary_key='id')
        t2.insert([{'id': 10, 'val': 'x'}, {'id': 20, 'val': 'y'}])

        result1 = _get_table_md(engine, 'test_pk_multi_good')
        result2 = _get_table_md(engine, 'test_pk_multi_bad')
        assert result1 is not None and result2 is not None
        tbl_id1, _ = result1
        tbl_id2, md2 = result2

        # Simulate pre-migration for both tables
        _simulate_pre_migration(engine, tbl_id1)
        # Re-downgrade is needed since first call sets it to 48, but second table also needs setup
        _drop_pk_index(engine, tbl_id2)
        _strip_primary_index_md(engine, tbl_id2)
        _downgrade_to_v48(engine)

        # Insert duplicate into the second table
        store_name2 = f'tbl_{tbl_id2.hex}'
        id_col_id = None
        val_col_id = None
        for col_md in md2['column_md'].values():
            if col_md.get('is_pk', False):
                id_col_id = col_md['id']
            elif col_md.get('col_type', {}).get('_classname') == 'StringType':
                val_col_id = col_md['id']
        assert id_col_id is not None and val_col_id is not None

        with engine.begin() as conn:
            conn.execute(
                sql.text(
                    f'INSERT INTO {store_name2} (rowid, v_min, v_max, col_{id_col_id}, col_{val_col_id}) '
                    f'VALUES (999, 0, {_MAX_VERSION}, 10, :val)'
                ),
                {'val': 'duplicate'},
            )

        # Trigger migration
        Env._init_env()
        reload_catalog()

        # Verify: good table has index and PrimaryIndexMd
        engine = Env.get().engine
        assert _index_exists(engine, tbl_id1), 'Good table should have PK index'
        _, md1 = _get_table_md(engine, 'test_pk_multi_good')
        assert md1.get('primary_index_md') is not None

        # Verify: bad table has NO index and PK stripped
        assert not _index_exists(engine, tbl_id2), 'Bad table should NOT have PK index'
        _, md2_after = _get_table_md(engine, 'test_pk_multi_bad')
        assert md2_after.get('primary_index_md') is None
        for col_md in md2_after['column_md'].values():
            assert col_md.get('is_pk', False) is False

        pxt.drop_table('test_pk_multi_good', force=True)
        pxt.drop_table('test_pk_multi_bad', force=True)
