"""Spatial predicate expression for index-accelerated geometry queries.

When a spatial index exists on the geometry column, the predicate is pushed down to PostGIS SQL.
Otherwise, it falls back to Python evaluation using Shapely.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from pixeltable.catalog.table_version import TableVersion


class SpatialOp(enum.Enum):
    INTERSECTS = 'intersects'
    CONTAINS = 'contains'
    WITHIN = 'within'
    DWITHIN = 'dwithin'


class SpatialPredicate(Expr):
    """A spatial predicate expression against a geometry column.

    Uses the spatial index for SQL pushdown when available, falls back to Shapely for Python evaluation.
    """

    spatial_op: SpatialOp
    distance: float | None  # only used for DWITHIN

    def __init__(self, col_ref: ColumnRef, other: Expr, op: SpatialOp, distance: float | None = None):
        super().__init__(ts.BoolType())
        if not col_ref.col_type.is_geometry_type():
            raise excs.Error(f'Spatial predicates require a Geometry column, got {col_ref.col_type}')
        if op == SpatialOp.DWITHIN and distance is None:
            raise excs.Error('st_dwithin() requires a distance parameter')
        self.spatial_op = op
        self.distance = distance
        self.components = [col_ref, other]
        self.id = self._create_id()

    def __repr__(self) -> str:
        op_name = f'st_{self.spatial_op.value}'
        if self.distance is not None:
            return f'{self.components[0]}.{op_name}({self.components[1]}, distance={self.distance})'
        return f'{self.components[0]}.{op_name}({self.components[1]})'

    def _id_attrs(self) -> list[tuple[str, Any]]:
        attrs = [*super()._id_attrs(), ('spatial_op', self.spatial_op.value)]
        if self.distance is not None:
            attrs.append(('distance', self.distance))
        return attrs

    def default_column_name(self) -> str:
        return f'st_{self.spatial_op.value}'

    def _find_spatial_idx(self) -> 'TableVersion.IndexInfo | None':
        """Find a spatial index on the referenced column, or None if none exists."""
        from pixeltable.index import SpatialIndex

        col_ref = self.components[0]
        assert isinstance(col_ref, ColumnRef)
        tbl = col_ref.tbl.get()
        if col_ref.col.qid not in tbl.idxs_by_col:
            return None
        candidates = [info for info in tbl.idxs_by_col[col_ref.col.qid] if isinstance(info.idx, SpatialIndex)]
        return candidates[0] if candidates else None

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        from pixeltable.index import SpatialIndex

        if not isinstance(self.components[1], Literal):
            # Non-literal geometry argument — cannot push to SQL
            return None

        idx_info = self._find_spatial_idx()
        if idx_info is None:
            # No spatial index — fall back to Python eval
            return None

        assert isinstance(idx_info.idx, SpatialIndex)
        literal_val = self.components[1].val
        if literal_val is None:
            return sql.literal(False)

        if self.spatial_op == SpatialOp.DWITHIN:
            assert self.distance is not None
            dist_expr = idx_info.idx.distance_clause(idx_info.val_col, literal_val)
            return dist_expr <= self.distance
        else:
            return idx_info.idx.spatial_predicate_clause(idx_info.val_col, self.spatial_op.value, literal_val)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        """Python fallback using Shapely when no spatial index is available."""
        from shapely.geometry import shape

        col_val = data_row[self.components[0].slot_idx]
        other_val = (
            self.components[1].val if isinstance(self.components[1], Literal) else data_row[self.components[1].slot_idx]
        )

        if col_val is None or other_val is None:
            data_row[self.slot_idx] = False
            return

        geom = shape(col_val)
        other = shape(other_val)

        if self.spatial_op == SpatialOp.INTERSECTS:
            data_row[self.slot_idx] = geom.intersects(other)
        elif self.spatial_op == SpatialOp.CONTAINS:
            data_row[self.slot_idx] = geom.contains(other)
        elif self.spatial_op == SpatialOp.WITHIN:
            data_row[self.slot_idx] = geom.within(other)
        elif self.spatial_op == SpatialOp.DWITHIN:
            assert self.distance is not None
            data_row[self.slot_idx] = geom.distance(other) <= self.distance

    def _as_dict(self) -> dict:
        d = {'spatial_op': self.spatial_op.value, **super()._as_dict()}
        if self.distance is not None:
            d['distance'] = self.distance
        return d

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> SpatialPredicate:
        assert len(components) == 2
        assert isinstance(components[0], ColumnRef)
        op = SpatialOp(d['spatial_op'])
        distance = d.get('distance')
        return cls(components[0], components[1], op=op, distance=distance)
