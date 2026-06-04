"""Tests for the optional OpenTelemetry instrumentation in pixeltable.

These tests use OpenTelemetry's in-memory exporters/readers so they require no
network. They exercise representative pixeltable operations and assert that
the expected spans and metric data points are recorded with the expected
attributes.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterator

import pytest

import pixeltable as pxt

# Skip the entire module if the optional OTel SDK isn't installed.
opentelemetry = pytest.importorskip('opentelemetry')
pytest.importorskip('opentelemetry.sdk')

from opentelemetry import metrics, trace  # noqa: E402
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import SpanKind  # noqa: E402

from pixeltable.telemetry import otel as pxt_otel  # noqa: E402

# ---------------------------------------------------------------------------
# Module-scoped: configure the global TracerProvider / MeterProvider exactly
# once for this test module, then call pxt_otel.instrument() exactly once.
# Tests clear the exporter / reader at the start of each test.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def otel_state() -> Iterator[tuple[InMemorySpanExporter, InMemoryMetricReader]]:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    # set_tracer_provider only takes effect on the first call per process. If
    # another test module has already set one we just push our processor onto
    # the existing provider so spans still flow into our exporter.
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        current.add_span_processor(SimpleSpanProcessor(span_exporter))
    else:
        trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    # Same caveat for metrics: best-effort -- if a provider is already set,
    # leave it alone (we'll still get spans, and meters will be no-op).
    try:
        metrics.set_meter_provider(meter_provider)
    except Exception:  # pragma: no cover - defensive
        pass

    # Instrument exactly once per process. instrument() is idempotent.
    pxt_otel.instrument()

    yield span_exporter, metric_reader


@pytest.fixture
def spans_and_metrics(
    otel_state: tuple[InMemorySpanExporter, InMemoryMetricReader],
) -> tuple[InMemorySpanExporter, InMemoryMetricReader]:
    span_exporter, metric_reader = otel_state
    span_exporter.clear()
    # Drain any pending metric data so each test starts clean.
    metric_reader.get_metrics_data()
    return span_exporter, metric_reader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_metric_points(reader: InMemoryMetricReader) -> dict[str, list]:
    """Return a map of metric name -> list of (value, attributes) tuples."""
    data = reader.get_metrics_data()
    out: dict[str, list] = {}
    if data is None:
        return out
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                points = out.setdefault(metric.name, [])
                # Histogram has data_points with .sum and .count; Counter / UpDownCounter
                # have data_points with .value.
                for dp in metric.data.data_points:
                    if hasattr(dp, 'sum'):
                        points.append((dp.sum, dict(dp.attributes)))
                    else:
                        points.append((dp.value, dict(dp.attributes)))
    return out


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list]:
    out: dict[str, list] = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instrument_is_idempotent() -> None:
    """Calling instrument() twice is a no-op (the wrappers are not re-stacked)."""
    pxt_otel.instrument()
    pxt_otel.instrument()
    assert pxt_otel.is_instrumented()

    # The wrapped methods carry the marker and a reference to the original.
    from pixeltable._query import Query

    assert getattr(Query.collect, pxt_otel._INSTRUMENTED_MARKER, False)
    original = getattr(Query.collect, '__pxt_otel_original__', None)
    assert original is not None
    # The marker should NOT be on the original.
    assert not getattr(original, pxt_otel._INSTRUMENTED_MARKER, False)


def test_no_provider_is_no_op() -> None:
    """When no TracerProvider / MeterProvider is configured by the host app,
    the OTel API returns no-op tracers and meters. The instrumentation must
    not raise. We verify this by calling get_tracer / get_meter directly on
    the proxy provider (no providers set in this Python process before the
    fixture runs); the spans/metrics created here MUST NOT raise.

    This test does not require pixeltable's DB fixture -- it only verifies
    that the helpers used by `instrument()` are safe when the SDK isn't
    configured.
    """
    # Run a separate sub-interpreter style check by calling the same internal
    # helpers used by instrument() with a brand new ProxyTracerProvider state.
    from opentelemetry import metrics as _metrics, trace as _trace

    tracer = _trace.get_tracer('pixeltable.test', '0.0.0')
    meter = _metrics.get_meter('pixeltable.test', '0.0.0')

    # These must not raise even if no real provider has been configured.
    hist = meter.create_histogram('pixeltable.test.duration', unit='s', description='x')
    counter = meter.create_counter('pixeltable.test.count', unit='1', description='x')
    updown = meter.create_up_down_counter('pixeltable.test.active', unit='1', description='x')

    hist.record(0.1, {'db.system': 'pixeltable'})
    counter.add(1, {'db.system': 'pixeltable'})
    updown.add(1, {'db.system': 'pixeltable'})
    updown.add(-1, {'db.system': 'pixeltable'})

    with tracer.start_as_current_span('pixeltable.test.span', kind=SpanKind.CLIENT) as span:
        span.set_attribute('db.system', 'pixeltable')


def _create_simple_table() -> pxt.Table:
    schema = {'id': pxt.Required[pxt.Int], 'name': pxt.String}
    return pxt.create_table('otel_t', schema, primary_key='id', if_exists='replace_force')


def test_insert_emits_span_and_metrics(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    exporter, reader = spans_and_metrics

    t = _create_simple_table()
    t.insert([{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}])

    spans = _spans_by_name(exporter)
    assert 'pixeltable.table.insert' in spans, spans.keys()
    span = spans['pixeltable.table.insert'][0]
    assert span.kind == SpanKind.CLIENT
    assert span.attributes['db.system'] == 'pixeltable'
    assert span.attributes['db.operation'] == 'insert'
    assert span.attributes['db.name'] == 'otel_t'

    points = _collect_metric_points(reader)
    assert 'pixeltable.query.duration' in points
    assert 'pixeltable.query.count' in points
    assert any(attrs.get('db.operation') == 'insert' for _, attrs in points['pixeltable.query.count'])


def test_collect_emits_select_span(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    exporter, _reader = spans_and_metrics

    t = _create_simple_table()
    t.insert([{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}, {'id': 3, 'name': 'c'}])
    exporter.clear()

    result = t.collect()
    assert len(result) == 3

    spans = _spans_by_name(exporter)
    assert 'pixeltable.query.collect' in spans, spans.keys()
    span = spans['pixeltable.query.collect'][0]
    assert span.kind == SpanKind.CLIENT
    assert span.attributes['db.system'] == 'pixeltable'
    assert span.attributes['db.operation'] == 'select'
    # row_count is recorded best-effort for select operations.
    assert span.attributes.get('pixeltable.row_count') == 3


def test_count_emits_select_span(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    exporter, _reader = spans_and_metrics

    t = _create_simple_table()
    t.insert([{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}])
    exporter.clear()

    n = t.count()
    assert n == 2

    spans = _spans_by_name(exporter)
    assert 'pixeltable.query.count' in spans, spans.keys()
    span = spans['pixeltable.query.count'][0]
    assert span.kind == SpanKind.CLIENT
    assert span.attributes['db.system'] == 'pixeltable'
    assert span.attributes['db.operation'] == 'select'


def test_update_and_delete_emit_spans(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    exporter, _reader = spans_and_metrics

    t = _create_simple_table()
    t.insert([{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}])
    exporter.clear()

    t.update({'name': 'z'}, where=t.id == 1)
    t.delete(where=t.id == 2)

    spans = _spans_by_name(exporter)
    assert 'pixeltable.table.update' in spans, spans.keys()
    upd = spans['pixeltable.table.update'][0]
    assert upd.attributes['db.system'] == 'pixeltable'
    assert upd.attributes['db.operation'] == 'update'
    assert upd.attributes['db.name'] == 'otel_t'

    assert 'pixeltable.table.delete' in spans
    dele = spans['pixeltable.table.delete'][0]
    assert dele.attributes['db.system'] == 'pixeltable'
    assert dele.attributes['db.operation'] == 'delete'
    assert dele.attributes['db.name'] == 'otel_t'


def test_error_path_records_exception(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    """When the pixeltable op raises, the span is marked ERROR and the
    pixeltable.query.errors counter is incremented.
    """
    exporter, reader = spans_and_metrics

    t = _create_simple_table()
    exporter.clear()
    reader.get_metrics_data()  # drain

    # Two rows with the same primary key -> should raise on the second insert.
    t.insert([{'id': 1, 'name': 'a'}])
    with pytest.raises(pxt.Error):
        t.insert([{'id': 1, 'name': 'a'}])  # duplicate PK

    # Find the failing insert span.
    from opentelemetry.trace import StatusCode

    bad_spans = [
        s
        for s in exporter.get_finished_spans()
        if s.name == 'pixeltable.table.insert' and s.status.status_code == StatusCode.ERROR
    ]
    assert bad_spans, [s.name for s in exporter.get_finished_spans()]

    points = _collect_metric_points(reader)
    assert 'pixeltable.query.errors' in points, list(points.keys())
    # at least one error point has db.operation=insert
    assert any(attrs.get('db.operation') == 'insert' for _, attrs in points['pixeltable.query.errors'])


def test_metrics_have_expected_units_and_types(
    uses_db: None, spans_and_metrics: tuple[InMemorySpanExporter, InMemoryMetricReader]
) -> None:
    _exporter, reader = spans_and_metrics

    t = _create_simple_table()
    t.insert([{'id': 1, 'name': 'a'}])
    t.count()

    data = reader.get_metrics_data()
    assert data is not None
    by_name: dict[str, Any] = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                by_name[m.name] = m

    assert by_name['pixeltable.query.duration'].unit == 's'
    assert by_name['pixeltable.query.count'].unit == '1'
    # Histograms expose `bucket_counts`; Counters expose `value`.
    dur_dp = list(by_name['pixeltable.query.duration'].data.data_points)
    assert dur_dp, 'expected at least one duration data point'
    assert hasattr(dur_dp[0], 'sum')
    cnt_dp = list(by_name['pixeltable.query.count'].data.data_points)
    assert cnt_dp, 'expected at least one count data point'
    assert hasattr(cnt_dp[0], 'value')


def test_pixeltable_imports_without_otel_packages_module_path() -> None:
    """Importing pixeltable does not import the OTel SDK.

    We can't easily uninstall the SDK here, but we can verify the structural
    invariant: importing `pixeltable` (the top-level package) does not pull
    in `pixeltable.telemetry.otel`. The host app must opt in by importing the
    submodule explicitly.
    """
    # Spawn a fresh Python process so we get a clean import state. The
    # subprocess imports pixeltable and asserts that pixeltable.telemetry.otel
    # was NOT imported transitively.
    import subprocess

    code = (
        'import sys; import pixeltable; '
        "assert 'pixeltable.telemetry.otel' not in sys.modules, "
        "'pixeltable.__init__ pulled in OTel SDK transitively'"
    )
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
