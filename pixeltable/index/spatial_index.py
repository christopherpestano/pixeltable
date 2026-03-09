"""Spatial index using PostGIS GiST for geometry columns."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.exprs as exprs
import pixeltable.type_system as ts

from .base import IndexBase

if TYPE_CHECKING:
    import pixeltable.catalog as catalog


class PostGISGeometry(sql.types.UserDefinedType):
    """Minimal SQLAlchemy type for PostGIS geometry columns.

    Handles serialization of GeoJSON dicts to/from PostGIS geometry values.
    """

    cache_ok = True

    def __init__(self, srid: int = 4326):
        self.srid = srid

    def get_col_spec(self, **kw: Any) -> str:
        return f'geometry(Geometry,{self.srid})'

    def bind_processor(self, dialect: Any) -> Any:
        def process(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, dict):
                return json.dumps(value)
            return value  # already a string

        return process

    def bind_expression(self, bindvalue: Any) -> Any:
        return sql.func.ST_SetSRID(sql.func.ST_GeomFromGeoJSON(bindvalue), self.srid)


class SpatialIndex(IndexBase):
    """Interface to PostGIS GiST spatial index.

    Creates a shadow PostGIS geometry column alongside the JSONB source column and indexes it
    with a GiST index for efficient spatial queries.
    """

    srid: int

    def __init__(self, srid: int = 4326):
        self.srid = srid

    def create_value_expr(self, c: 'catalog.Column') -> 'exprs.Expr':
        if not c.col_type.is_geometry_type():
            raise excs.Error(f'Spatial index requires a Geometry column, but column {c.name!r} has type {c.col_type}')
        # The value expression is the identity — it passes through the GeoJSON dict.
        # The PostGISGeometry SQLAlchemy type handles the conversion to PostGIS geometry via bind_expression.
        return exprs.ColumnRef(c)

    def records_value_errors(self) -> bool:
        return False

    def get_index_sa_type(self, val_col_type: ts.ColumnType) -> sql.types.TypeEngine:
        return PostGISGeometry(srid=self.srid)

    def sa_create_stmt(self, store_index_name: str, sa_value_col: sql.Column) -> sql.Compiled:
        from sqlalchemy.dialects import postgresql

        sa_idx = sql.Index(store_index_name, sa_value_col, postgresql_using='gist')
        return sql.schema.CreateIndex(sa_idx, if_not_exists=True).compile(dialect=postgresql.dialect())

    def spatial_predicate_clause(
        self, val_col: 'catalog.Column', op: str, literal_geojson: dict[str, Any]
    ) -> sql.ColumnElement:
        """Generate a PostGIS SQL predicate for the given spatial operation.

        Args:
            val_col: The index value column (PostGIS geometry type).
            op: Spatial operation name ('intersects', 'contains', 'within', 'dwithin').
            literal_geojson: GeoJSON dict for the query geometry.

        Returns:
            A SQLAlchemy column element representing the spatial predicate.
        """
        geojson_str = json.dumps(literal_geojson)
        other_geom = sql.func.ST_SetSRID(sql.func.ST_GeomFromGeoJSON(geojson_str), self.srid)
        col = val_col.sa_col

        if op == 'intersects':
            return sql.func.ST_Intersects(col, other_geom)
        elif op == 'contains':
            return sql.func.ST_Contains(col, other_geom)
        elif op == 'within':
            return sql.func.ST_Within(col, other_geom)
        else:
            raise excs.Error(f'Unknown spatial operation: {op!r}')

    def distance_clause(self, val_col: 'catalog.Column', literal_geojson: dict[str, Any]) -> sql.ColumnElement:
        """Generate a PostGIS SQL expression for distance computation.

        Args:
            val_col: The index value column (PostGIS geometry type).
            literal_geojson: GeoJSON dict for the query geometry.

        Returns:
            A SQLAlchemy column element representing the distance.
        """
        geojson_str = json.dumps(literal_geojson)
        other_geom = sql.func.ST_SetSRID(sql.func.ST_GeomFromGeoJSON(geojson_str), self.srid)
        return sql.func.ST_Distance(val_col.sa_col, other_geom)

    @classmethod
    def display_name(cls) -> str:
        return 'spatial'

    def as_dict(self) -> dict:
        return {'srid': self.srid}

    @classmethod
    def from_dict(cls, d: dict) -> SpatialIndex:
        return cls(srid=d.get('srid', 4326))
