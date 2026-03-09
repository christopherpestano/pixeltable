import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, shape

import pixeltable as pxt
import pixeltable.type_system as ts


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
