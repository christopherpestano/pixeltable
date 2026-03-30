import csv
import os
import pathlib

import pixeltable as pxt
from pixeltable.io.utils import normalize_media_url

from ..utils import create_all_datatypes_tbl, create_test_tbl, get_audio_files, get_image_files, validate_update_status


class TestCsv:
    def test_export_round_trip(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Export a table to CSV, re-import, and verify equality."""
        t = create_test_tbl('test_csv_rt')

        csv_path = tmp_path / 'round_trip.csv'
        # Select only columns whose types survive CSV round-trip (string, int, float, bool)
        query = t.select(t.c1, t.c1n, t.c2, t.c3, t.c4)
        pxt.io.export_csv(query, csv_path)

        t2 = pxt.io.import_csv('test_csv_rt_reimported', str(csv_path))

        assert query.collect() == t2.collect()

    def test_export_with_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Nulls become empty strings in CSV."""
        t = pxt.create_table(
            'test_csv_nulls', {'c_int': pxt.Int, 'c_string': pxt.String, 'c_float': pxt.Float, 'c_json': pxt.Json}
        )
        t.insert(
            [
                {'c_int': 1, 'c_string': None, 'c_float': None, 'c_json': None},
                {'c_int': None, 'c_string': 'hello', 'c_float': 1.5, 'c_json': {'a': 1}},
            ]
        )

        csv_path = tmp_path / 'nulls.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))

        assert exported[0]['c_string'] == ''
        assert exported[0]['c_float'] == ''
        assert exported[0]['c_json'] == ''
        assert exported[1]['c_int'] == ''

    def test_export_with_query(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test export with filtering and column selection."""
        t = pxt.create_table('test_csv_query', {'c_int': pxt.Int, 'c_string': pxt.String})
        validate_update_status(t.insert([{'c_int': i, 'c_string': f'row_{i}'} for i in range(10)]), expected_rows=10)

        csv_path = tmp_path / 'filtered.csv'
        pxt.io.export_csv(t.where(t.c_int < 5), csv_path)
        with open(csv_path, encoding='utf-8') as f:
            assert len(list(csv.DictReader(f))) == 5

        csv_path2 = tmp_path / 'subset.csv'
        pxt.io.export_csv(t.select(t.c_string), csv_path2)
        with open(csv_path2, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))
        assert len(exported) == 10
        assert list(exported[0].keys()) == ['c_string']

    def test_export_custom_delimiter(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Test CSV export with a tab delimiter."""
        t = pxt.create_table('test_csv_delim', {'c_int': pxt.Int, 'c_string': pxt.String})
        t.insert([{'c_int': 1, 'c_string': 'hello'}, {'c_int': 2, 'c_string': 'world'}])

        csv_path = tmp_path / 'tab.csv'
        pxt.io.export_csv(t, csv_path, delimiter='\t')

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f, delimiter='\t'))
        assert len(exported) == 2
        assert int(exported[0]['c_int']) == 1
        assert exported[1]['c_string'] == 'world'

    def test_export_media_urls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Media columns export original local paths, never ~/.pixeltable cache paths."""
        t = create_all_datatypes_tbl()

        csv_path = tmp_path / 'media.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))

        assert len(exported) > 0

        media_cols = ['c_image', 'c_video', 'c_audio', 'c_document']
        for row in exported:
            for col in media_cols:
                val = row[col]
                assert '.pixeltable' not in val, f'Exported {col} contains .pixeltable cache path: {val}'

        original_image = get_image_files()[0]
        assert exported[0]['c_image'] == original_image

        audio_files = get_audio_files()
        assert exported[0]['c_audio'] in audio_files

        for col in media_cols:
            val = exported[0][col]
            assert os.path.exists(val), f'Exported {col} path does not exist: {val}'

        csv_path2 = tmp_path / 'media_select.csv'
        pxt.io.export_csv(t.select(t.c_image, t.c_string), csv_path2)

        with open(csv_path2, encoding='utf-8') as f:
            exported2 = list(csv.DictReader(f))

        assert list(exported2[0].keys()) == ['c_image', 'c_string']
        for row in exported2:
            assert '.pixeltable' not in row['c_image'], (
                f'Selected c_image contains .pixeltable cache path: {row["c_image"]}'
            )
            assert row['c_image'] == original_image

    def test_export_media_urls_with_nulls(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Null media columns export as empty string in CSV."""
        t = pxt.create_table('test_csv_media_nulls', {'c_name': pxt.String, 'c_image': pxt.Image})
        original_image = get_image_files()[0]
        t.insert([{'c_name': 'has_image', 'c_image': original_image}, {'c_name': 'no_image', 'c_image': None}])

        csv_path = tmp_path / 'media_nulls.csv'
        pxt.io.export_csv(t, csv_path)

        with open(csv_path, encoding='utf-8') as f:
            exported = list(csv.DictReader(f))

        row_with = next(r for r in exported if r['c_name'] == 'has_image')
        assert row_with['c_image'] == original_image
        assert '.pixeltable' not in row_with['c_image']

        row_without = next(r for r in exported if r['c_name'] == 'no_image')
        assert row_without['c_image'] == ''

    def test_normalize_media_url(self) -> None:
        """normalize_media_url converts file:// URLs to paths and passes remote URLs through."""
        assert normalize_media_url('file:///tmp/test/image.jpg') == '/tmp/test/image.jpg'
        assert normalize_media_url('file:///tmp/test%20dir/image.jpg') == '/tmp/test dir/image.jpg'
        assert normalize_media_url('https://example.com/image.jpg') == 'https://example.com/image.jpg'
        assert normalize_media_url('s3://bucket/key.jpg') == 's3://bucket/key.jpg'
        assert normalize_media_url(None) is None
