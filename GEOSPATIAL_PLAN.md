# Geospatial Support Implementation Plan

## Phase 1: Core Type System + JSONB Storage + Basic UDFs + GeoDataFrame Import

### 1A. `GeometryType` in the type system

**File: `pixeltable/type_system.py`**

- Add `GEOMETRY = 14` and `RASTER = 15` to `ColumnType.Type` enum (after `BINARY = 13`)
- `GEOMETRY` = vector data (points, lines, polygons). Implemented in Phase 1.
- `RASTER` = geospatial raster/gridded data (satellite imagery, elevation, land cover). Enum reserved now; implementation deferred to a future phase (requires rasterio).
- Create `GeometryType(ColumnType)` with optional `geom_type` parameter
- Create user-facing `Geometry(_PxtType)` following `Json` pattern (no first parent class)
- Update `make_type()`, `infer_literal_type()`, `from_python_type()`
- Add `is_geometry_type()` helper, add to `ALL_PIXELTABLE_TYPES`
- Export `Geometry` from `pixeltable/__init__.py`

**Storage**: JSONB (GeoJSON dicts). Shapely objects normalized to GeoJSON in `_create_literal()`.

### 1B. GeoDataFrame import/export

**New file: `pixeltable/io/geodataframe.py`**

- `import_geodataframe()` — thin wrapper around `create_table()` + `insert()`
- `gdf_infer_schema()` — delegates non-geometry cols to `df_infer_schema()`, maps geometry cols to `GeometryType`
- `_gdf_row_to_pxt_row()` — converts shapely → GeoJSON via `mapping()`
- Add `ResultSet.to_geodataframe()` in `pixeltable/_query.py`
- Register GeoDataFrame in `TableDataSource` union in `pixeltable/globals.py`

### 1C. Geometry UDFs

**New file: `pixeltable/functions/geometry.py`**

Method UDFs receive `self` as a dict (GeoJSON from JSONB). Each converts via `shapely.geometry.shape()`.

- **Unary methods**: buffer, centroid, convex_hull, simplify
- **Properties**: area, length, is_valid, geom_type, bounds
- **Binary methods**: distance, intersection, union, difference, contains, intersects, within
- **Conversion**: from_geojson, to_wkt, from_wkt

### 1D. Dependencies

`pyproject.toml`: `geo = ["shapely>=2.0", "geopandas>=0.14"]`

### 1E. Tests + Validation

**New file: `tests/test_geometry.py`**

| Action | File |
|--------|------|
| Modify | `pixeltable/type_system.py` |
| Modify | `pixeltable/__init__.py` |
| Modify | `pixeltable/globals.py` |
| Modify | `pixeltable/_query.py` |
| Modify | `pixeltable/io/__init__.py` |
| Modify | `pyproject.toml` |
| Create | `pixeltable/io/geodataframe.py` |
| Create | `pixeltable/functions/geometry.py` |
| Create | `tests/test_geometry.py` |

---

## Phase 2: Notebook Visualization with Interactive Maps

### 2A. Geometry formatting

**File: `pixeltable/utils/formatter.py`**

- `format_geometry()` renders inline Leaflet map via folium
- Hook into `_format_cell()` — detect GeometryType BEFORE Json/dict fallback
- Threshold: >20 rows → truncated JSON instead of iframe maps
- Graceful fallback to JSON text if folium not installed

### 2B. Standalone map helper

**New file: `pixeltable/utils/map.py`**

- `show_map(table, geom_col, ...)` → `folium.Map` with all geometries
- Export as `pxt.show_map()` (not `pxt.map()` — shadows builtin)

| Action | File |
|--------|------|
| Modify | `pixeltable/utils/formatter.py` |
| Create | `pixeltable/utils/map.py` |
| Modify | `pixeltable/__init__.py` |

---

## Phase 3: PostGIS Spatial Indexing + Server-Side Query Pushdown

### 3A. PostGIS detection

**New file: `pixeltable/utils/postgis.py`** — `postgis_available()`, `ensure_postgis()`

### 3B. Spatial index

**New file: `pixeltable/index/spatial_index.py`** — `SpatialIndex(IndexBase)` with GiST

### 3C. Storage: dual-column strategy

When `add_spatial_index()` is called, add a shadow PostGIS `geometry` column alongside JSONB. Spatial index is on the PostGIS column. Read path unchanged.

### 3D. Spatial predicate expression

**New file: `pixeltable/exprs/spatial_predicate.py`** — `SpatialPredicate(Expr)` following `SimilarityExpr` pattern. Query geometry must be a Literal. Falls back to shapely Python eval when no index.

### 3E. Table API

`table.add_spatial_index(col_name, srid=4326)` in `catalog/table.py`

| Action | File |
|--------|------|
| Create | `pixeltable/utils/postgis.py` |
| Create | `pixeltable/index/spatial_index.py` |
| Create | `pixeltable/exprs/spatial_predicate.py` |
| Modify | `pixeltable/catalog/table.py` |
| Modify | `pixeltable/exec/sql_node.py` |
| Modify | `pixeltable/plan.py` |
| Modify | `pyproject.toml` (`geoalchemy2>=0.15`) |

---

## Phase 4: Iterators + Advanced Spatial Operations

### 4A. Geometry iterator

**New file: `pixeltable/iterators/geometry.py`** — `geometry_splitter` with `explode` and `grid` modes

### 4B. Advanced spatial UDFs

Extend `pixeltable/functions/geometry.py`: `transform` (CRS reprojection), `voronoi_polygons`

### 4C. Spatial joins (stretch goal)

Deferred until join infrastructure matures.

| Action | File |
|--------|------|
| Create | `pixeltable/iterators/geometry.py` |
| Modify | `pixeltable/functions/geometry.py` |
| Modify | `pyproject.toml` (`pyproj>=3.6`) |

---

## Dependencies

| Package | Phase | Extra group |
|---------|-------|-------------|
| `shapely>=2.0` | 1 | `geo` |
| `geopandas>=0.14` | 1 | `geo` |
| `folium>=0.15` | 2 | `geo` |
| `geoalchemy2>=0.15` | 3 | `geo-postgis` |
| `pyproj>=3.6` | 4 | `geo` |

## Risks

| Risk | Phase | Mitigation |
|------|-------|------------|
| JSONB bloat for complex polygons | 1 | Phase 3 PostGIS resolves |
| CRS lost in GeoJSON | 1 | Store SRID in GeometryType; default EPSG:4326 |
| Folium iframes heavy in large results | 2 | Threshold ~20 rows |
| Dual-column doubles storage | 3 | Only when spatial index added |
| Literal constraint blocks col-vs-col predicates | 3 | Python fallback; joins in Phase 4 |
| geopandas import slow (~2s) | 1 | Lazy import |
