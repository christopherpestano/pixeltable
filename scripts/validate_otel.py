#!/usr/bin/env python3
"""Manual validation script for Pixeltable's OpenTelemetry instrumentation.

This script configures the OpenTelemetry SDK with stdout exporters, calls
``pixeltable.telemetry.otel.instrument()``, exercises a handful of pixeltable
operations on a small in-memory table, and prints a summary.

Run::

    python scripts/validate_otel.py

The script exits 0 on success and prints at least one span and one metric data
point to stdout. ``--help`` prints usage and exits 0.

This is intended for local sanity-checking and CI smoke tests. The actual
correctness assertions live in ``tests/test_otel.py``.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from pathlib import Path


def _configure_otel() -> tuple:
    """Configure the OTel SDK with stdout exporters.

    Returns the (tracer_provider, meter_provider, captured_stdout) tuple. The
    captured_stdout is an ``io.StringIO`` that the Console exporters write
    into so we can later assert "at least one span / one metric was emitted"
    without polluting the user's terminal more than necessary.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    import pixeltable

    resource = Resource.create(
        {'service.name': 'pixeltable', 'service.version': getattr(pixeltable, '__version__', '')}
    )

    captured = io.StringIO()

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=captured)))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter(out=captured), export_interval_millis=500)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, meter_provider, captured


def _exercise_pixeltable() -> None:
    """Run a handful of pixeltable operations against a fresh local instance."""
    import pixeltable as pxt

    # Insert a couple of rows, run a query, an update, and a delete -- one of
    # each operation kind we instrument.
    table_name = 'otel_validation_tbl'
    t = pxt.create_table(
        table_name, {'id': pxt.Required[pxt.Int], 'name': pxt.String}, primary_key='id', if_exists='replace_force'
    )
    t.insert([{'id': 1, 'name': 'alice'}, {'id': 2, 'name': 'bob'}, {'id': 3, 'name': 'carol'}])
    rows = t.collect()
    print(f'validate_otel: collected {len(rows)} rows')
    n = t.count()
    print(f'validate_otel: count() == {n}')
    t.update({'name': 'updated'}, where=t.id == 1)
    t.delete(where=t.id == 3)
    pxt.drop_table(table_name, if_not_exists='ignore')


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='validate_otel.py',
        description=(
            'Configure OTel with stdout exporters, instrument pixeltable, run a few '
            'operations against a temporary local Pixeltable instance, and print a summary.'
        ),
        epilog=(
            'On success the script exits 0 and prints at least one span and one metric '
            'data point. This is intended for local sanity-checking only; correctness '
            'assertions live in tests/test_otel.py.'
        ),
    )
    parser.add_argument(
        '--keep-home',
        action='store_true',
        help='Do not delete the temporary $PIXELTABLE_HOME on exit (useful for debugging).',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Use an ephemeral home so we don't touch the user's real pixeltable instance.
    tmp_home = Path(tempfile.mkdtemp(prefix='pxt-otel-validate-'))
    os.environ['PIXELTABLE_HOME'] = str(tmp_home)
    os.environ.setdefault('PIXELTABLE_START_DASHBOARD', 'false')

    try:
        import pixeltable as pxt

        pxt.init()

        tracer_provider, meter_provider, captured = _configure_otel()

        from pixeltable.telemetry import otel as pxt_otel

        pxt_otel.instrument()

        _exercise_pixeltable()

        # Give the periodic metric reader a chance to flush.
        time.sleep(0.6)
        # Force-flush so we don't have to wait for the periodic interval.
        try:
            tracer_provider.force_flush(timeout_millis=2000)
        except Exception:
            pass
        try:
            meter_provider.force_flush(timeout_millis=2000)
        except Exception:
            pass

        captured_text = captured.getvalue()
        # Print captured exporter output so the operator can eyeball the
        # spans / metric data points.
        print('\n===== captured OTel exporter output =====')
        print(captured_text)
        print('===== end captured output =====\n')

        has_span = 'pixeltable.table.insert' in captured_text or 'pixeltable.query.collect' in captured_text
        has_metric = 'pixeltable.query.duration' in captured_text or 'pixeltable.query.count' in captured_text

        if not has_span:
            print('validate_otel: ERROR - no pixeltable spans were emitted', file=sys.stderr)
            return 2
        if not has_metric:
            print('validate_otel: ERROR - no pixeltable metric data points were emitted', file=sys.stderr)
            return 3

        print('validate_otel: OK -- at least one span and one metric data point emitted')
        return 0
    finally:
        if not args.keep_home:
            import shutil

            shutil.rmtree(tmp_home, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
