from __future__ import annotations

from typing import TYPE_CHECKING, Any

import folium
from shapely.geometry import shape

import pixeltable.type_system as ts

if TYPE_CHECKING:
    from pixeltable.catalog import Table


def show_map(
    table: Table,
    geom_col: str,
    *,
    limit: int | None = None,
    style: dict[str, Any] | None = None,
    tooltip_cols: list[str] | None = None,
    zoom_start: int = 12,
) -> folium.Map:
    """Render all geometries from a table column on an interactive folium map.

    Args:
        table: The Pixeltable table to read from.
        geom_col: Name of the geometry column.
        limit: Maximum number of rows to render. ``None`` renders all rows.
        style: Optional GeoJSON style dict passed to ``folium.GeoJson``.
        tooltip_cols: Column names to show in hover tooltips.
        zoom_start: Initial zoom level for the map.

    Returns:
        A ``folium.Map`` that renders in Jupyter notebooks.
    """
    col_ref = getattr(table, geom_col)
    schema = table._get_schema()
    if not isinstance(schema.get(geom_col), ts.GeometryType):
        raise ValueError(f'Column {geom_col!r} is not a Geometry column')

    select_cols = [col_ref]
    extra_names: list[str] = []
    if tooltip_cols:
        for name in tooltip_cols:
            select_cols.append(getattr(table, name))
            extra_names.append(name)

    query = table.select(*select_cols)
    if limit is not None:
        query = query.limit(limit)
    result = query.collect()

    features: list[dict[str, Any]] = []
    for i in range(len(result)):
        row = result[i]
        geojson = row[geom_col]
        if geojson is None:
            continue
        props = {name: row[name] for name in extra_names}
        features.append({'type': 'Feature', 'geometry': geojson, 'properties': props})

    fc = {'type': 'FeatureCollection', 'features': features}

    m = folium.Map(zoom_start=zoom_start)

    tooltip = folium.GeoJsonTooltip(fields=extra_names) if extra_names else None
    folium.GeoJson(fc, style_function=lambda x, s=style: s or {}, tooltip=tooltip).add_to(m)

    if features:
        all_bounds = [shape(f['geometry']).bounds for f in features]
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m
