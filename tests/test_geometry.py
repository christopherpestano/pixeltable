import folium
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box, shape

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.utils.formatter import Formatter


class TestGeometryType:
    def test_type_basics(self) -> None:
        gt = ts.GeometryType()
        assert gt.is_geometry_type()
        assert not gt.is_json_type()
        assert gt.geom_type is None

        gt_point = ts.GeometryType(geom_type='point')
        assert gt_point.geom_type == 'POINT'

        gt_poly = ts.GeometryType(geom_type='Polygon')
        assert gt_poly.geom_type == 'POLYGON'

        with pytest.raises(pxt.Error, match='Invalid geometry type'):
            ts.GeometryType(geom_type='INVALID')

    def test_matches_and_supertype(self) -> None:
        gt1 = ts.GeometryType()
        gt2 = ts.GeometryType()
        assert gt1.matches(gt2)

        gt_point = ts.GeometryType(geom_type='POINT')
        gt_poly = ts.GeometryType(geom_type='POLYGON')
        assert not gt_point.matches(gt_poly)

        # supertype of two different geom_types is unconstrained
        st = gt_point.supertype(gt_poly)
        assert st is not None
        assert st.geom_type is None

        # supertype of same geom_type preserves it
        st = gt_point.supertype(ts.GeometryType(geom_type='POINT'))
        assert st is not None
        assert st.geom_type == 'POINT'

    def test_serialization_roundtrip(self) -> None:
        for gt in [ts.GeometryType(), ts.GeometryType(geom_type='MULTIPOLYGON', nullable=True)]:
            d = gt.as_dict()
            restored = ts.ColumnType.from_dict(d)
            assert restored == gt

    def test_validate_literal_shapely(self) -> None:
        gt = ts.GeometryType()
        gt.validate_literal(Point(0, 0))
        gt.validate_literal(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        gt.validate_literal(MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]))

    def test_validate_literal_geojson(self) -> None:
        gt = ts.GeometryType()
        gt.validate_literal({'type': 'Point', 'coordinates': [0, 0]})
        gt.validate_literal({'type': 'GeometryCollection', 'geometries': []})

        with pytest.raises(TypeError, match='type'):
            gt.validate_literal({'coordinates': [0, 0]})
        with pytest.raises(TypeError, match='coordinates'):
            gt.validate_literal({'type': 'Point'})
        with pytest.raises(TypeError):
            gt.validate_literal(42)

    def test_create_literal_normalizes_shapely(self) -> None:
        gt = ts.GeometryType()
        result = gt.create_literal(Point(1.0, 2.0))
        assert isinstance(result, dict)
        assert result['type'] == 'Point'
        assert result['coordinates'] == (1.0, 2.0)

    def test_infer_literal_type(self) -> None:
        ct = ts.ColumnType.infer_literal_type(Point(0, 0))
        assert ct is not None
        assert ct.is_geometry_type()

    def test_pxt_type_parameterization(self) -> None:
        # unparameterized
        ct = pxt.Geometry.as_col_type(nullable=False)
        assert isinstance(ct, ts.GeometryType)
        assert ct.geom_type is None

        # parameterized
        annotated = pxt.Geometry['POINT']  # type: ignore[name-defined]
        ct = ts.ColumnType.from_python_type(annotated)
        assert isinstance(ct, ts.GeometryType)
        assert ct.geom_type == 'POINT'


class TestGeometryTable:
    def test_create_and_insert(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo', {'geom': pxt.Geometry, 'name': pxt.String})
        point_geojson = {'type': 'Point', 'coordinates': [1.0, 2.0]}
        poly_geojson = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}

        t.insert([{'geom': point_geojson, 'name': 'point'}, {'geom': poly_geojson, 'name': 'polygon'}])
        result = t.collect()
        assert len(result) == 2
        rows = [result[i] for i in range(len(result))]
        names = {r['name'] for r in rows}
        assert names == {'point', 'polygon'}

    def test_insert_shapely_objects(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_shapely', {'geom': pxt.Geometry, 'label': pxt.String})
        t.insert(
            [
                {'geom': Point(10, 20), 'label': 'a'},
                {'geom': Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]), 'label': 'b'},
            ]
        )
        result = t.collect()
        assert len(result) == 2
        # stored as GeoJSON dicts
        for i in range(len(result)):
            assert isinstance(result[i]['geom'], dict)
            assert 'type' in result[i]['geom']

    def test_nullable_geometry(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_null', {'geom': pxt.Geometry, 'val': pxt.Int})
        t.insert([{'geom': Point(0, 0), 'val': 1}, {'geom': None, 'val': 2}])
        result = t.collect()
        assert len(result) == 2

    def test_parameterized_geometry(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_point', {'geom': pxt.Geometry['POINT'], 'val': pxt.Int})  # type: ignore[misc]
        t.insert([{'geom': Point(1, 2), 'val': 1}])
        result = t.collect()
        assert len(result) == 1


class TestGeometryUDFs:
    def test_unary_methods(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_udf', {'geom': pxt.Geometry})
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        t.insert([{'geom': poly}])

        t.add_computed_column(buffered=t.geom.buffer(1.0))
        t.add_computed_column(hull=t.geom.convex_hull())
        t.add_computed_column(simplified=t.geom.simplify(0.5))
        t.add_computed_column(env=t.geom.envelope())
        t.add_computed_column(ctr=t.geom.centroid())

        result = t.collect()
        row = result[0]
        assert row['buffered'] is not None
        assert row['hull'] is not None
        assert row['simplified'] is not None
        assert row['env'] is not None
        ctr = Point(*shape(row['ctr']).coords[0])
        assert abs(ctr.x - 5.0) < 1e-6
        assert abs(ctr.y - 5.0) < 1e-6

    def test_properties(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_props', {'geom': pxt.Geometry})
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        t.insert([{'geom': poly}])

        # use computed columns so we control the names
        t.add_computed_column(area_val=t.geom.area)
        t.add_computed_column(length_val=t.geom.length)
        t.add_computed_column(valid=t.geom.is_valid)
        t.add_computed_column(gtype=t.geom.geom_type)
        t.add_computed_column(bbox=t.geom.bounds)

        result = t.collect()
        row = result[0]
        assert abs(row['area_val'] - 100.0) < 1e-6
        assert row['valid'] is True
        assert row['gtype'] == 'Polygon'
        assert len(row['bbox']) == 4

    def test_binary_methods(self, uses_db: None) -> None:
        t = pxt.create_table('test_geo_binary', {'a': pxt.Geometry, 'b': pxt.Geometry})
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        p2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)])
        t.insert([{'a': p1, 'b': p2}])

        t.add_computed_column(dist=t.a.distance(t.b))
        t.add_computed_column(inter=t.a.intersection(t.b))
        t.add_computed_column(uni=t.a.union(t.b))
        t.add_computed_column(diff=t.a.difference(t.b))
        t.add_computed_column(does_intersect=t.a.intersects(t.b))
        t.add_computed_column(does_contain=t.a.contains(t.b))
        t.add_computed_column(is_within=t.a.within(t.b))

        result = t.collect()
        row = result[0]
        assert row['dist'] == 0.0  # overlapping
        assert row['does_intersect'] is True
        assert row['does_contain'] is False
        assert row['is_within'] is False
        inter_area = shape(row['inter']).area
        assert abs(inter_area - 25.0) < 1e-6

    def test_conversion_udfs(self, uses_db: None) -> None:
        import pixeltable.functions.geometry as geo

        t = pxt.create_table('test_geo_convert', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(1, 2)}])

        t.add_computed_column(wkt=t.geom.to_wkt())

        result = t.collect()
        assert result[0]['wkt'] == 'POINT (1 2)'

        # from_wkt
        t2 = pxt.create_table('test_geo_from_wkt', {'wkt': pxt.String})
        t2.insert([{'wkt': 'POINT (3 4)'}])
        t2.add_computed_column(geom=geo.from_wkt(t2.wkt))
        result = t2.collect()
        assert result[0]['geom']['type'] == 'Point'
        assert list(result[0]['geom']['coordinates']) == [3.0, 4.0]


class TestGeometryImportExport:
    def test_geodataframe_roundtrip(self, uses_db: None) -> None:
        import geopandas
        from shapely.geometry import Point

        gdf = geopandas.GeoDataFrame(
            {'name': ['a', 'b', 'c'], 'value': [1, 2, 3], 'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]}
        )

        t = pxt.io.import_geodataframe('test_gdf', gdf)
        assert len(t.collect()) == 3

        result_gdf = t.collect().to_geodataframe()
        assert isinstance(result_gdf, geopandas.GeoDataFrame)
        assert len(result_gdf) == 3
        assert result_gdf.geometry.name == 'geometry'
        for geom in result_gdf.geometry:
            assert geom.geom_type == 'Point'


class TestGeometryVisualization:
    def test_format_geometry_renders_map(self) -> None:
        fmt = Formatter(num_rows=1, num_cols=1, http_address='http://localhost')
        geojson = {'type': 'Point', 'coordinates': [1.0, 2.0]}
        result = fmt.format_geometry(geojson)
        assert 'pxt_geometry' in result
        assert '<iframe' in result

    def test_format_geometry_falls_back_for_large_results(self) -> None:
        fmt = Formatter(num_rows=25, num_cols=1, http_address='http://localhost')
        geojson = {'type': 'Point', 'coordinates': [1.0, 2.0]}
        html = fmt.format_geometry(geojson)
        assert 'pxt_geometry' not in html

    def test_format_geometry_handles_none(self) -> None:
        fmt = Formatter(num_rows=1, num_cols=1, http_address='http://localhost')
        assert fmt.format_geometry(None) == ''

    def test_show_map(self, uses_db: None) -> None:
        t = pxt.create_table('test_show_map', {'geom': pxt.Geometry, 'name': pxt.String})
        t.insert(
            [
                {'geom': Point(0, 0), 'name': 'origin'},
                {'geom': Point(1, 1), 'name': 'one'},
                {'geom': Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]), 'name': 'square'},
            ]
        )
        m = pxt.show_map(t, 'geom', tooltip_cols=['name'])
        assert isinstance(m, folium.Map)

    def test_show_map_with_limit(self, uses_db: None) -> None:
        t = pxt.create_table('test_show_map_limit', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(i, i)} for i in range(10)])
        m = pxt.show_map(t, 'geom', limit=3)
        assert isinstance(m, folium.Map)

    def test_show_map_rejects_non_geometry(self, uses_db: None) -> None:
        t = pxt.create_table('test_show_map_err', {'val': pxt.Int})
        t.insert([{'val': 1}])
        with pytest.raises(ValueError, match='not a Geometry column'):
            pxt.show_map(t, 'val')

    def test_result_set_show_map(self, uses_db: None) -> None:
        t = pxt.create_table('test_rs_map', {'geom': pxt.Geometry, 'name': pxt.String})
        t.insert(
            [
                {'geom': Point(0, 0), 'name': 'origin'},
                {'geom': Point(1, 1), 'name': 'one'},
                {'geom': Point(2, 2), 'name': 'two'},
            ]
        )
        # select a subset and show on one map
        m = t.where(t.name.isin(['origin', 'one'])).collect().show_map()
        assert isinstance(m, folium.Map)

    def test_result_set_show_map_auto_tooltips(self, uses_db: None) -> None:
        t = pxt.create_table('test_rs_map_tt', {'geom': pxt.Geometry, 'label': pxt.String, 'val': pxt.Int})
        t.insert([{'geom': Point(0, 0), 'label': 'a', 'val': 1}])
        m = t.collect().show_map()
        assert isinstance(m, folium.Map)


class TestPostGISDetection:
    def test_postgis_available_returns_bool(self, uses_db: None) -> None:
        from pixeltable.utils.postgis import postgis_available, reset_cache

        reset_cache()
        result = postgis_available()
        assert isinstance(result, bool)
        # calling again returns the cached value
        assert postgis_available() == result

    def test_ensure_postgis_raises_when_unavailable(self, uses_db: None) -> None:
        from pixeltable.utils.postgis import ensure_postgis, postgis_available

        if postgis_available():
            pytest.skip('PostGIS is available, cannot test unavailable path')
        with pytest.raises(RuntimeError, match='PostGIS is not available'):
            ensure_postgis()

    def test_add_spatial_index_raises_when_no_postgis(self, uses_db: None) -> None:
        from pixeltable.utils.postgis import postgis_available

        if postgis_available():
            pytest.skip('PostGIS is available, cannot test unavailable path')
        t = pxt.create_table('test_no_postgis', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(0, 0)}])
        with pytest.raises(RuntimeError, match='PostGIS is not available'):
            t.add_spatial_index('geom')


class TestSpatialPredicateFallback:
    """Tests for spatial predicates using Python/Shapely fallback (no spatial index required)."""

    def test_st_intersects_python_fallback(self, uses_db: None) -> None:
        t = pxt.create_table('test_sp_intersects', {'geom': pxt.Geometry, 'name': pxt.String})
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        p2 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)])
        t.insert([{'geom': p1, 'name': 'overlapping'}, {'geom': p2, 'name': 'distant'}])

        query_poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)])
        result = t.where(t.geom.st_intersects(query_poly)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'overlapping'

    def test_st_contains_python_fallback(self, uses_db: None) -> None:
        t = pxt.create_table('test_sp_contains', {'geom': pxt.Geometry, 'name': pxt.String})
        big = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        small = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        t.insert([{'geom': big, 'name': 'big'}, {'geom': small, 'name': 'small'}])

        inner_point = {'type': 'Point', 'coordinates': [50, 50]}
        result = t.where(t.geom.st_contains(inner_point)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'big'

    def test_st_within_python_fallback(self, uses_db: None) -> None:
        t = pxt.create_table('test_sp_within', {'geom': pxt.Geometry, 'name': pxt.String})
        t.insert([{'geom': Point(5, 5), 'name': 'inside'}, {'geom': Point(50, 50), 'name': 'outside'}])

        bounding_box = box(0, 0, 10, 10)
        result = t.where(t.geom.st_within(bounding_box)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'inside'

    def test_st_dwithin_python_fallback(self, uses_db: None) -> None:
        t = pxt.create_table('test_sp_dwithin', {'geom': pxt.Geometry, 'name': pxt.String})
        t.insert([{'geom': Point(1, 1), 'name': 'near'}, {'geom': Point(100, 100), 'name': 'far'}])

        origin = Point(0, 0)
        result = t.where(t.geom.st_dwithin(origin, distance=5.0)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'near'

    def test_spatial_predicate_handles_null(self, uses_db: None) -> None:
        t = pxt.create_table('test_sp_null', {'geom': pxt.Geometry, 'val': pxt.Int})
        t.insert([{'geom': Point(0, 0), 'val': 1}, {'geom': None, 'val': 2}])

        query_poly = box(-10, -10, 10, 10)
        result = t.where(t.geom.st_intersects(query_poly)).collect()
        assert len(result) == 1
        assert result[0]['val'] == 1

    def test_spatial_predicate_rejects_non_geometry(self) -> None:
        """Spatial predicates should raise an error on non-geometry columns."""
        from pixeltable.exprs.literal import Literal
        from pixeltable.exprs.spatial_predicate import SpatialOp, SpatialPredicate

        # Create a literal with a non-geometry type to test the validation
        with pytest.raises(pxt.Error, match='Geometry column'):
            SpatialPredicate(
                Literal('hello', col_type=ts.StringType()),  # type: ignore[arg-type]
                Literal({'type': 'Point', 'coordinates': [0, 0]}, col_type=ts.GeometryType()),
                op=SpatialOp.INTERSECTS,
            )

    def test_spatial_predicate_with_geojson_dict(self, uses_db: None) -> None:
        """Verify that GeoJSON dicts work as query geometries."""
        t = pxt.create_table('test_sp_geojson', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(5, 5)}])

        geojson_box = {'type': 'Polygon', 'coordinates': [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
        result = t.where(t.geom.st_within(geojson_box)).collect()
        assert len(result) == 1


class TestSpatialIndex:
    """Tests for PostGIS spatial index creation and query pushdown.

    These tests require PostGIS and are skipped when not available.
    """

    def _skip_without_postgis(self) -> None:
        from pixeltable.utils.postgis import postgis_available

        if not postgis_available():
            pytest.skip('PostGIS not available')

    def test_add_and_drop_spatial_index(self, uses_db: None) -> None:
        self._skip_without_postgis()
        t = pxt.create_table('test_si_basic', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(1, 2)}, {'geom': Point(3, 4)}])
        t.add_spatial_index('geom')

        indexes = t.list_indexes()
        assert len(indexes) == 1
        assert indexes[0]['_column'] == 'geom'

        t.drop_spatial_index(column='geom')
        assert len(t.list_indexes()) == 0

    def test_spatial_index_accelerated_intersects(self, uses_db: None) -> None:
        self._skip_without_postgis()
        t = pxt.create_table('test_si_intersects', {'geom': pxt.Geometry, 'name': pxt.String})
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        p2 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30), (20, 20)])
        t.insert([{'geom': p1, 'name': 'near'}, {'geom': p2, 'name': 'far'}])
        t.add_spatial_index('geom')

        query_poly = Polygon([(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)])
        result = t.where(t.geom.st_intersects(query_poly)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'near'

    def test_spatial_index_accelerated_within(self, uses_db: None) -> None:
        self._skip_without_postgis()
        t = pxt.create_table('test_si_within', {'geom': pxt.Geometry, 'name': pxt.String})
        t.insert([{'geom': Point(5, 5), 'name': 'inside'}, {'geom': Point(50, 50), 'name': 'outside'}])
        t.add_spatial_index('geom')

        bounding_box = box(0, 0, 10, 10)
        result = t.where(t.geom.st_within(bounding_box)).collect()
        assert len(result) == 1
        assert result[0]['name'] == 'inside'

    def test_spatial_index_if_exists(self, uses_db: None) -> None:
        self._skip_without_postgis()
        t = pxt.create_table('test_si_exists', {'geom': pxt.Geometry})
        t.insert([{'geom': Point(0, 0)}])
        t.add_spatial_index('geom', idx_name='my_idx')

        # duplicate should raise
        with pytest.raises(pxt.Error, match='Duplicate index name'):
            t.add_spatial_index('geom', idx_name='my_idx')

        # ignore should not raise
        t.add_spatial_index('geom', idx_name='my_idx', if_exists='ignore')

    def test_spatial_index_non_geometry_raises(self, uses_db: None) -> None:
        self._skip_without_postgis()
        t = pxt.create_table('test_si_bad_col', {'val': pxt.Int})
        t.insert([{'val': 1}])
        with pytest.raises(pxt.Error, match='Geometry column'):
            t.add_spatial_index('val')
