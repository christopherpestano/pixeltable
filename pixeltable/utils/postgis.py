"""PostGIS detection and setup utilities."""

from __future__ import annotations

import logging

import sqlalchemy as sql

_logger = logging.getLogger('pixeltable')

_cache: dict[str, bool] = {}


def postgis_available() -> bool:
    """Check if the PostGIS extension is available in the current database server.

    The result is cached after the first call.
    """
    if 'available' in _cache:
        return _cache['available']

    from pixeltable.env import Env

    engine = Env.get()._sa_engine
    with engine.begin() as conn:
        row = conn.execute(sql.text("SELECT 1 FROM pg_available_extensions WHERE name = 'postgis'")).fetchone()
    _cache['available'] = row is not None
    return _cache['available']


def ensure_postgis() -> None:
    """Create the PostGIS extension if not already installed.

    Raises:
        RuntimeError: If PostGIS is not available on the database server.
    """
    if not postgis_available():
        raise RuntimeError(
            'PostGIS is not available on this PostgreSQL server. '
            'Spatial indexing requires PostGIS. Install PostGIS or connect to a PostgreSQL server with PostGIS enabled.'
        )

    from pixeltable.env import Env

    engine = Env.get()._sa_engine
    with engine.begin() as conn:
        conn.execute(sql.text('CREATE EXTENSION IF NOT EXISTS postgis'))
    _logger.info('PostGIS extension enabled')


def reset_cache() -> None:
    """Reset the PostGIS availability cache. Useful for testing."""
    _cache.clear()
