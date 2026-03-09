"""Benchmark: PostGIS spatial index vs Shapely Python fallback for intersection queries."""

import random
import time

import pytest
from shapely.geometry import box

import pixeltable as pxt


def _random_polygon(x_center: float, y_center: float, size: float = 0.5) -> dict:
    """Generate a small GeoJSON polygon around a center point."""
    half = size / 2
    coords = [
        [x_center - half, y_center - half],
        [x_center + half, y_center - half],
        [x_center + half, y_center + half],
        [x_center - half, y_center + half],
        [x_center - half, y_center - half],
    ]
    return {'type': 'Polygon', 'coordinates': [coords]}


def _postgis_available() -> bool:
    from pixeltable.utils.postgis import postgis_available

    return postgis_available()


class TestSpatialPerformance:
    """Compare PostGIS index-accelerated vs Shapely fallback intersection queries."""

    @pytest.mark.parametrize('row_count', [100, 1000, 5000])
    def test_intersection_postgis_vs_shapely(self, uses_db: None, row_count: int) -> None:
        """Benchmark: PostGIS spatial index vs Shapely fallback for st_intersects.

        Creates two identical tables with the same data. One gets a spatial index
        (PostGIS pushdown), the other does not (Shapely fallback). Measures query time
        for both and prints the comparison.
        """
        if not _postgis_available():
            pytest.skip('PostGIS not available')

        random.seed(42)
        rows = [{'geom': _random_polygon(random.uniform(-180, 180), random.uniform(-90, 90))} for _ in range(row_count)]

        # Table WITH spatial index (PostGIS pushdown)
        t_indexed = pxt.create_table(f'bench_postgis_{row_count}', {'geom': pxt.Geometry})
        t_indexed.insert(rows)
        t_indexed.add_spatial_index('geom')

        # Table WITHOUT spatial index (Shapely fallback)
        t_fallback = pxt.create_table(f'bench_shapely_{row_count}', {'geom': pxt.Geometry})
        t_fallback.insert(rows)

        # Query polygon covering ~10% of the coordinate space
        query_poly = box(-18, -9, 18, 9)

        n_iterations = 5

        # Warm up
        t_indexed.where(t_indexed.geom.st_intersects(query_poly)).collect()
        t_fallback.where(t_fallback.geom.st_intersects(query_poly)).collect()

        # Benchmark PostGIS
        postgis_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result_postgis = t_indexed.where(t_indexed.geom.st_intersects(query_poly)).collect()
            postgis_times.append(time.perf_counter() - start)

        # Benchmark Shapely fallback
        shapely_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            result_shapely = t_fallback.where(t_fallback.geom.st_intersects(query_poly)).collect()
            shapely_times.append(time.perf_counter() - start)

        # Verify both return the same number of results
        assert len(result_postgis) == len(result_shapely), (
            f'Result mismatch: PostGIS={len(result_postgis)}, Shapely={len(result_shapely)}'
        )

        avg_postgis = sum(postgis_times) / n_iterations
        avg_shapely = sum(shapely_times) / n_iterations
        speedup = avg_shapely / avg_postgis if avg_postgis > 0 else float('inf')

        print(f'\n--- Spatial Intersection Benchmark ({row_count} rows, {len(result_postgis)} matches) ---')
        print(f'  PostGIS (indexed):    {avg_postgis * 1000:.2f} ms avg ({n_iterations} runs)')
        print(f'  Shapely (fallback):   {avg_shapely * 1000:.2f} ms avg ({n_iterations} runs)')
        print(f'  Speedup:              {speedup:.2f}x')
