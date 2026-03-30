"""Edge-case tests for media-URL export (resolve_media_to_urls / normalize_media_url)."""

import csv
import json
import pathlib

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.io.utils import normalize_media_url

from ..utils import get_audio_files, get_documents, get_image_files, get_video_files, validate_update_status


def _safe_document_path() -> str:
    """Return a document path that works regardless of whether mistune is installed."""
    docs = get_documents()
    if not Env.get().is_installed_package('mistune'):
        mistune_exts = {'.md', '.pptx', '.docx', '.xlsx'}
        docs = [d for d in docs if pathlib.Path(d).suffix.lower() not in mistune_exts]
    assert len(docs) > 0, 'No usable document files found in test data'
    return docs[0]


class TestNormalizeMediaUrl:
    """Direct unit tests for the normalize_media_url helper."""

    def test_none_passthrough(self) -> None:
        assert normalize_media_url(None) is None

    def test_remote_http_passthrough(self) -> None:
        url = 'https://example.com/images/photo.jpg'
        assert normalize_media_url(url) == url

    def test_remote_https_passthrough(self) -> None:
        url = 'https://cdn.example.org/video.mp4'
        assert normalize_media_url(url) == url

    def test_s3_passthrough(self) -> None:
        url = 's3://my-bucket/path/to/file.pdf'
        assert normalize_media_url(url) == url

    def test_gs_passthrough(self) -> None:
        url = 'gs://my-bucket/audio/clip.flac'
        assert normalize_media_url(url) == url

    def test_file_url_converted_to_path(self) -> None:
        assert normalize_media_url('file:///home/user/img.jpg') == '/home/user/img.jpg'

    def test_file_url_with_encoded_spaces(self) -> None:
        result = normalize_media_url('file:///home/user/my%20photos/img.jpg')
        assert result == '/home/user/my photos/img.jpg'

    def test_file_url_with_encoded_special_chars(self) -> None:
        result = normalize_media_url('file:///tmp/data%231/file%25name.png')
        assert result == '/tmp/data#1/file%name.png'

    def test_empty_string(self) -> None:
        """Empty string is not None and does not start with 'file:', so it passes through."""
        assert normalize_media_url('') == ''

    def test_file_url_windows_style(self) -> None:
        """A file:// URL with a drive letter (Windows convention)."""
        # On Linux url2pathname will return /C:/Users/... which is fine as a passthrough test
        result = normalize_media_url('file:///C:/Users/test/image.png')
        assert 'C:' in result
        assert 'image.png' in result

    def test_data_url_passthrough(self) -> None:
        """Data URLs should pass through unchanged (unlikely but possible)."""
        url = 'data:image/png;base64,iVBORw0KGgo='
        assert normalize_media_url(url) == url


class TestMediaExportEdgeCases:
    """Integration tests exercising export_csv / export_json with media edge cases."""

    def test_csv_no_media_columns(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """A table with zero media columns should export without error."""
        t = pxt.create_table('no_media_csv', {'name': pxt.String, 'score': pxt.Int})
        validate_update_status(t.insert([{'name': 'alice', 'score': 42}, {'name': 'bob', 'score': 99}]))
        out = tmp_path / 'no_media.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]['name'] == 'alice'
        assert rows[1]['score'] == '99'

    def test_json_no_media_columns(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        t = pxt.create_table('no_media_json', {'name': pxt.String, 'score': pxt.Int})
        validate_update_status(t.insert([{'name': 'alice', 'score': 42}]))
        out = tmp_path / 'no_media.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 1
        assert rows[0]['name'] == 'alice'
        assert rows[0]['score'] == 42

    def test_csv_only_media_columns(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """All columns are media -- every column should be URL-swapped."""
        img = get_image_files()[0]
        audio = get_audio_files()[0]
        t = pxt.create_table('only_media_csv', {'img': pxt.Image, 'aud': pxt.Audio})
        validate_update_status(t.insert([{'img': img, 'aud': audio}]))
        out = tmp_path / 'only_media.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        for col in ['img', 'aud']:
            val = rows[0][col]
            assert isinstance(val, str)
            assert len(val) > 0, f'{col} should not be empty'

    def test_json_only_media_columns(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        img = get_image_files()[0]
        vid = get_video_files()[0]
        t = pxt.create_table('only_media_json', {'img': pxt.Image, 'vid': pxt.Video})
        validate_update_status(t.insert([{'img': img, 'vid': vid}]))
        out = tmp_path / 'only_media.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 1
        for col in ['img', 'vid']:
            assert isinstance(rows[0][col], str)
            assert len(rows[0][col]) > 0

    def test_csv_null_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Null media values should become empty strings in CSV."""
        t = pxt.create_table('null_media_csv', {'img': pxt.Image, 'label': pxt.String})
        validate_update_status(t.insert([{'img': None, 'label': 'no-image'}]))
        out = tmp_path / 'null_media.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['img'] == ''
        assert rows[0]['label'] == 'no-image'

    def test_json_null_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Null media values should become None (JSON null)."""
        t = pxt.create_table('null_media_json', {'img': pxt.Image, 'label': pxt.String})
        validate_update_status(t.insert([{'img': None, 'label': 'no-image'}]))
        out = tmp_path / 'null_media.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert rows[0]['img'] is None
        assert rows[0]['label'] == 'no-image'

    def test_csv_mixed_null_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Some rows have media, others do not."""
        img = get_image_files()[0]
        t = pxt.create_table('mixed_null_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(
            t.insert([{'img': img, 'idx': 0}, {'img': None, 'idx': 1}, {'img': img, 'idx': 2}]), expected_rows=3
        )
        out = tmp_path / 'mixed_null.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert len(rows[0]['img']) > 0
        assert rows[1]['img'] == ''
        assert len(rows[2]['img']) > 0

    def test_json_mixed_null_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        img = get_image_files()[0]
        t = pxt.create_table('mixed_null_json', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(
            t.insert([{'img': img, 'idx': 0}, {'img': None, 'idx': 1}, {'img': img, 'idx': 2}]), expected_rows=3
        )
        out = tmp_path / 'mixed_null.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 3
        assert isinstance(rows[0]['img'], str) and len(rows[0]['img']) > 0
        assert rows[1]['img'] is None
        assert isinstance(rows[2]['img'], str) and len(rows[2]['img']) > 0

    def test_csv_where_clause_preserved(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Filtering should still work after the URL swap."""
        img = get_image_files()[0]
        t = pxt.create_table('where_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': i} for i in range(5)]), expected_rows=5)
        out = tmp_path / 'where.csv'
        pxt.io.export_csv(t.where(t.idx >= 3), out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        idxs = sorted(int(r['idx']) for r in rows)
        assert idxs == [3, 4]

    def test_json_limit_preserved(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """limit() should be preserved after the URL swap."""
        img = get_image_files()[0]
        t = pxt.create_table('limit_json', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': i} for i in range(10)]), expected_rows=10)
        out = tmp_path / 'limit.json'
        pxt.io.export_json(t.order_by(t.idx).limit(3), out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 3

    def test_csv_order_by_preserved(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """order_by should be preserved so the exported rows come out in the right order."""
        img = get_image_files()[0]
        t = pxt.create_table('order_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': i} for i in range(5)]), expected_rows=5)
        out = tmp_path / 'order.csv'
        pxt.io.export_csv(t.order_by(t.idx, asc=False), out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        idxs = [int(r['idx']) for r in rows]
        assert idxs == [4, 3, 2, 1, 0]

    def test_json_where_and_order_combined(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Combine where + order_by + limit and verify all are preserved."""
        img = get_image_files()[0]
        t = pxt.create_table('combo_json', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': i} for i in range(10)]), expected_rows=10)
        out = tmp_path / 'combo.json'
        pxt.io.export_json(t.where(t.idx >= 2).order_by(t.idx, asc=False).limit(3), out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 3
        idxs = [r['idx'] for r in rows]
        assert idxs == [9, 8, 7]

    def test_csv_select_subset_only_swaps_selected_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """When selecting a subset of columns, only selected media columns appear and are swapped."""
        img = get_image_files()[0]
        audio = get_audio_files()[0]
        t = pxt.create_table('select_sub_csv', {'img': pxt.Image, 'aud': pxt.Audio, 'label': pxt.String})
        validate_update_status(t.insert([{'img': img, 'aud': audio, 'label': 'test'}]))
        out = tmp_path / 'select_sub.csv'
        pxt.io.export_csv(t.select(t.img, t.label), out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == ['img', 'label']
        assert len(rows[0]['img']) > 0
        assert rows[0]['label'] == 'test'

    def test_json_select_non_media_only(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Selecting only non-media columns: media columns should not appear at all."""
        img = get_image_files()[0]
        t = pxt.create_table('select_nomedia_json', {'img': pxt.Image, 'label': pxt.String})
        validate_update_status(t.insert([{'img': img, 'label': 'test'}]))
        out = tmp_path / 'select_nomedia.json'
        pxt.io.export_json(t.select(t.label), out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert list(rows[0].keys()) == ['label']
        assert rows[0]['label'] == 'test'

    def test_csv_all_four_media_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Table with Image, Video, Audio, and Document columns."""
        img = get_image_files()[0]
        vid = get_video_files()[0]
        aud = get_audio_files()[0]
        doc = _safe_document_path()

        t = pxt.create_table(
            'four_media_csv', {'img': pxt.Image, 'vid': pxt.Video, 'aud': pxt.Audio, 'doc': pxt.Document}
        )
        validate_update_status(t.insert([{'img': img, 'vid': vid, 'aud': aud, 'doc': doc}]))
        out = tmp_path / 'four_media.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        for col in ['img', 'vid', 'aud', 'doc']:
            val = rows[0][col]
            assert isinstance(val, str) and len(val) > 0, f'Expected non-empty string for {col}, got {val!r}'
            assert not val.startswith('file:'), f'{col} still has a file:// URL: {val}'

    def test_json_all_four_media_types(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        img = get_image_files()[0]
        vid = get_video_files()[0]
        aud = get_audio_files()[0]
        doc = _safe_document_path()

        t = pxt.create_table(
            'four_media_json', {'img': pxt.Image, 'vid': pxt.Video, 'aud': pxt.Audio, 'doc': pxt.Document}
        )
        validate_update_status(t.insert([{'img': img, 'vid': vid, 'aud': aud, 'doc': doc}]))
        out = tmp_path / 'four_media.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 1
        for col in ['img', 'vid', 'aud', 'doc']:
            val = rows[0][col]
            assert isinstance(val, str) and len(val) > 0, f'{col} should be non-empty string, got {val!r}'
            assert not val.startswith('file:'), f'{col} still has a file:// URL: {val}'

    def test_csv_multiple_rows_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Verify that media URLs are correctly resolved for every row, not just the first."""
        images = get_image_files()[:3]
        t = pxt.create_table('multi_row_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': images[i], 'idx': i} for i in range(3)]), expected_rows=3)
        out = tmp_path / 'multi_row.csv'
        pxt.io.export_csv(t.order_by(t.idx), out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        paths = [r['img'] for r in rows]
        for p in paths:
            assert isinstance(p, str) and len(p) > 0
        if images[0] != images[2]:
            assert paths[0] != paths[2], 'Different input images should produce different URLs'

    def test_json_media_with_json_and_array_columns(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Ensure media swapping does not interfere with JSON or array column serialization."""
        import numpy as np

        img = get_image_files()[0]
        t = pxt.create_table(
            'media_complex_json',
            {'img': pxt.Image, 'meta': pxt.Json, 'arr': pxt.Array[(3,), pxt.Float], 'label': pxt.String},
        )
        validate_update_status(
            t.insert(
                [
                    {
                        'img': img,
                        'meta': {'key': 'value', 'nested': [1, 2, 3]},
                        'arr': np.array([1.0, 2.0, 3.0], dtype=np.float32),
                        'label': 'test',
                    }
                ]
            )
        )
        out = tmp_path / 'media_complex.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        row = rows[0]
        assert isinstance(row['img'], str) and len(row['img']) > 0
        assert row['meta'] == {'key': 'value', 'nested': [1, 2, 3]}
        assert row['arr'] == [1.0, 2.0, 3.0]
        assert row['label'] == 'test'

    def test_csv_table_direct_export(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Pass a Table object directly (not a Query) to export_csv."""
        img = get_image_files()[0]
        t = pxt.create_table('direct_tbl_csv', {'img': pxt.Image, 'val': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'val': 1}]))
        out = tmp_path / 'direct.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert len(rows[0]['img']) > 0
        assert not rows[0]['img'].startswith('file:')

    def test_json_table_direct_export(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Pass a Table object directly (not a Query) to export_json."""
        img = get_image_files()[0]
        t = pxt.create_table('direct_tbl_json', {'img': pxt.Image, 'val': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'val': 1}]))
        out = tmp_path / 'direct.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 1
        assert isinstance(rows[0]['img'], str) and len(rows[0]['img']) > 0

    def test_csv_all_media_null(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Every media column is NULL for all rows."""
        t = pxt.create_table('all_null_media_csv', {'img': pxt.Image, 'vid': pxt.Video, 'label': pxt.String})
        validate_update_status(
            t.insert([{'img': None, 'vid': None, 'label': 'r0'}, {'img': None, 'vid': None, 'label': 'r1'}]),
            expected_rows=2,
        )
        out = tmp_path / 'all_null.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for r in rows:
            assert r['img'] == ''
            assert r['vid'] == ''

    def test_json_all_media_null(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        t = pxt.create_table('all_null_media_json', {'img': pxt.Image, 'aud': pxt.Audio, 'label': pxt.String})
        validate_update_status(
            t.insert([{'img': None, 'aud': None, 'label': 'r0'}, {'img': None, 'aud': None, 'label': 'r1'}]),
            expected_rows=2,
        )
        out = tmp_path / 'all_null.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert len(rows) == 2
        for r in rows:
            assert r['img'] is None
            assert r['aud'] is None

    def test_exported_csv_paths_exist_on_disk(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """For local files the exported paths should actually exist on disk."""
        img = get_image_files()[0]
        t = pxt.create_table('path_exists_csv', {'img': pxt.Image})
        validate_update_status(t.insert([{'img': img}]))
        out = tmp_path / 'path_exists.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        exported_path = rows[0]['img']
        assert pathlib.Path(exported_path).exists(), f'Exported path does not exist: {exported_path}'

    def test_exported_json_paths_exist_on_disk(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        img = get_image_files()[0]
        t = pxt.create_table('path_exists_json', {'img': pxt.Image})
        validate_update_status(t.insert([{'img': img}]))
        out = tmp_path / 'path_exists.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        exported_path = rows[0]['img']
        assert pathlib.Path(exported_path).exists(), f'Exported path does not exist: {exported_path}'

    def test_csv_empty_table_with_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """An empty table with media columns should export headers but no data rows."""
        t = pxt.create_table('empty_media_csv', {'img': pxt.Image, 'label': pxt.String})
        out = tmp_path / 'empty.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)
        assert 'img' in header
        assert 'label' in header
        assert len(data_rows) == 0

    def test_json_empty_table_with_media(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        t = pxt.create_table('empty_media_json', {'img': pxt.Image, 'label': pxt.String})
        out = tmp_path / 'empty.json'
        pxt.io.export_json(t, out)
        with open(out, encoding='utf-8') as f:
            rows = json.load(f)
        assert rows == []

    def test_csv_duplicate_media_paths(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Same image path inserted in multiple rows should produce identical exported URLs."""
        img = get_image_files()[0]
        t = pxt.create_table('dup_media_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': 0}, {'img': img, 'idx': 1}]), expected_rows=2)
        out = tmp_path / 'dup.csv'
        pxt.io.export_csv(t, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert rows[0]['img'] == rows[1]['img']

    def test_csv_query_object(self, uses_db: None, tmp_path: pathlib.Path) -> None:
        """Explicitly pass a Query object rather than Table to export_csv."""
        img = get_image_files()[0]
        t = pxt.create_table('query_obj_csv', {'img': pxt.Image, 'idx': pxt.Int})
        validate_update_status(t.insert([{'img': img, 'idx': i} for i in range(3)]), expected_rows=3)
        query = t.select(t.img, t.idx).where(t.idx > 0).order_by(t.idx)
        out = tmp_path / 'query_obj.csv'
        pxt.io.export_csv(query, out)
        with open(out, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert int(rows[0]['idx']) == 1
        assert int(rows[1]['idx']) == 2
        for r in rows:
            assert len(r['img']) > 0
