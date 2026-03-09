from __future__ import annotations

import os
from typing import Any

import geopandas
from shapely.geometry import mapping as _mapping

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.io.pandas import df_infer_schema


def import_geodataframe(
    tbl_name: str,
    gdf: geopandas.GeoDataFrame,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> pxt.Table:
    """Creates a new table from a GeoDataFrame.

    Geometry columns are stored as GeoJSON (JSONB). Non-geometry columns are
    inferred using the standard pandas schema inference.

    Args:
        tbl_name: The name of the table to create.
        gdf: The GeoDataFrame to import.
        schema_overrides: Optional type overrides keyed by column name.
        primary_key: Column(s) to use as primary key.

    Returns:
        A handle to the newly created table.
    """
    schema = gdf_infer_schema(gdf, schema_overrides, primary_key)

    rows: list[dict[str, Any]] = []
    for _, row in gdf.iterrows():
        pxt_row: dict[str, Any] = {}
        for col_name in schema:
            val = row[col_name]
            if isinstance(schema[col_name], ts.GeometryType) and val is not None:
                pxt_row[col_name] = _mapping(val)
            else:
                pxt_row[col_name] = val
        rows.append(pxt_row)

    t = pxt.create_table(
        tbl_name,
        schema,  # type: ignore[arg-type]
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )
    t.insert(rows)
    return t


def import_geofile(
    tbl_name: str,
    filepath: str | os.PathLike,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """Creates a new table from a geospatial file.

    Reads shapefiles, GeoJSON, GeoPackage, KML, file geodatabases, and any
    other format supported by GDAL/OGR via ``geopandas.read_file()``.

    Args:
        tbl_name: The name of the table to create.
        filepath: Path or URL to the geospatial file.
        schema_overrides: Optional type overrides keyed by column name.
        primary_key: Column(s) to use as primary key.
        **kwargs: Additional arguments passed to ``geopandas.read_file()``.

    Returns:
        A handle to the newly created table.
    """
    gdf = geopandas.read_file(filepath, **kwargs)
    return import_geodataframe(
        tbl_name,
        gdf,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def gdf_infer_schema(
    gdf: geopandas.GeoDataFrame,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
) -> dict[str, ts.ColumnType]:
    """Infer a Pixeltable schema from a GeoDataFrame.

    Delegates non-geometry columns to the pandas schema inference, then
    overrides geometry columns with GeometryType.
    """
    if schema_overrides is None:
        schema_overrides = {}
    if primary_key is None:
        primary_key = []
    elif isinstance(primary_key, str):
        primary_key = [primary_key]

    # Pre-populate geometry columns so df_infer_schema doesn't choke on the geometry dtype
    for col_name in gdf.columns:
        if col_name in schema_overrides:
            continue
        if isinstance(gdf[col_name].dtype, geopandas.array.GeometryDtype):
            geom_types = gdf[col_name].dropna().geom_type.unique()
            geom_type = geom_types[0].upper() if len(geom_types) == 1 else None
            schema_overrides[col_name] = ts.GeometryType(geom_type=geom_type)

    return df_infer_schema(gdf, schema_overrides, primary_key)
