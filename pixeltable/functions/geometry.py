from typing import Any

from shapely import from_wkt as _from_wkt
from shapely.geometry import mapping as _mapping, shape as _shape
from shapely.geometry.base import BaseGeometry

import pixeltable as pxt


def _to_shapely(val: Any) -> BaseGeometry:
    return _shape(val)


def _to_geojson(geom: BaseGeometry) -> dict[str, Any]:
    return _mapping(geom)


# Unary methods


@pxt.udf(is_method=True)
def buffer(self: pxt.Geometry, distance: float) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).buffer(distance))


@pxt.udf(is_method=True)
def centroid(self: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).centroid)


@pxt.udf(is_method=True)
def convex_hull(self: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).convex_hull)


@pxt.udf(is_method=True)
def simplify(self: pxt.Geometry, tolerance: float) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).simplify(tolerance))


@pxt.udf(is_method=True)
def envelope(self: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).envelope)


# Unary properties


@pxt.udf(is_property=True)
def area(self: pxt.Geometry) -> float:
    return _to_shapely(self).area


@pxt.udf(is_property=True)
def length(self: pxt.Geometry) -> float:
    return _to_shapely(self).length


@pxt.udf(is_property=True)
def is_valid(self: pxt.Geometry) -> bool:
    return _to_shapely(self).is_valid


@pxt.udf(is_property=True)
def geom_type(self: pxt.Geometry) -> str:
    return _to_shapely(self).geom_type


@pxt.udf(is_property=True)
def bounds(self: pxt.Geometry) -> pxt.Json:
    return list(_to_shapely(self).bounds)


# Binary methods


@pxt.udf(is_method=True)
def distance(self: pxt.Geometry, other: pxt.Geometry) -> float:
    return _to_shapely(self).distance(_to_shapely(other))


@pxt.udf(is_method=True)
def intersection(self: pxt.Geometry, other: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).intersection(_to_shapely(other)))


@pxt.udf(is_method=True)
def union(self: pxt.Geometry, other: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).union(_to_shapely(other)))


@pxt.udf(is_method=True)
def difference(self: pxt.Geometry, other: pxt.Geometry) -> pxt.Geometry:
    return _to_geojson(_to_shapely(self).difference(_to_shapely(other)))


@pxt.udf(is_method=True)
def contains(self: pxt.Geometry, other: pxt.Geometry) -> bool:
    return _to_shapely(self).contains(_to_shapely(other))


@pxt.udf(is_method=True)
def intersects(self: pxt.Geometry, other: pxt.Geometry) -> bool:
    return _to_shapely(self).intersects(_to_shapely(other))


@pxt.udf(is_method=True)
def within(self: pxt.Geometry, other: pxt.Geometry) -> bool:
    return _to_shapely(self).within(_to_shapely(other))


# Conversion functions


@pxt.udf
def from_geojson(geojson: pxt.Json) -> pxt.Geometry:
    _to_shapely(geojson)  # validate
    return geojson


@pxt.udf(is_method=True)
def to_wkt(self: pxt.Geometry) -> str:
    return _to_shapely(self).wkt


@pxt.udf
def from_wkt(wkt: str) -> pxt.Geometry:
    return _to_geojson(_from_wkt(wkt))
