"""OpenTelemetry instrumentation for Pixeltable.

This module is **optional**. It is safe to import even when the OpenTelemetry
SDK is not installed -- the SDK is only imported inside :func:`instrument`,
which raises a clear ``ImportError`` if the dependencies are missing.

When :func:`instrument` is called without a configured ``TracerProvider`` /
``MeterProvider``, the OpenTelemetry API returns no-op tracers and meters, so
the instrumentation adds no observable overhead and emits no telemetry. The
host application is responsible for configuring the providers.

Spans, metrics, attributes, and units are defined by
``otel-spec.md`` (committed alongside the PR that introduced this module).
Highlights:

- All spans are kind ``CLIENT`` with ``db.system="pixeltable"``.
- Span names: ``pixeltable.query.collect``, ``pixeltable.query.count``,
  ``pixeltable.table.insert``, ``pixeltable.table.update``,
  ``pixeltable.table.batch_update``, ``pixeltable.table.delete``.
- Metrics (unit, type): ``pixeltable.query.duration`` (s, Histogram),
  ``pixeltable.query.count`` (1, Counter),
  ``pixeltable.query.errors`` (1, Counter),
  ``pixeltable.query.active`` (1, UpDownCounter).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable

_LOG = logging.getLogger(__name__)

# Public marker used on wrapped functions so instrument() is idempotent.
_INSTRUMENTED_MARKER = '__pxt_otel_instrumented__'

# Standard semconv keys and values used across the instrumentation.
_DB_SYSTEM_KEY = 'db.system'
_DB_SYSTEM_VALUE = 'pixeltable'
_DB_OPERATION_KEY = 'db.operation'
_DB_NAME_KEY = 'db.name'

# Operation tokens used in span attributes and metric attributes.
_OP_SELECT = 'select'
_OP_INSERT = 'insert'
_OP_UPDATE = 'update'
_OP_DELETE = 'delete'

# Module-level state populated by instrument().
_INSTRUMENT_LOCK = threading.Lock()
_INSTRUMENTED = False
_TRACER: Any = None
_METER: Any = None
_DURATION_HISTOGRAM: Any = None
_COUNT_COUNTER: Any = None
_ERROR_COUNTER: Any = None
_ACTIVE_UPDOWN: Any = None


def _require_otel() -> tuple[Any, Any]:
    """Import the OTel API modules, raising a friendly error if missing."""
    try:
        from opentelemetry import metrics, trace
    except ImportError as e:
        raise ImportError(
            "OpenTelemetry is not installed. Install pixeltable with the 'otel' "
            "extra to enable telemetry: pip install 'pixeltable[otel]'."
        ) from e
    return trace, metrics


def _build_instruments() -> None:
    """Create the tracer, meter, and instruments. Idempotent."""
    global _TRACER, _METER  # noqa: PLW0603
    global _DURATION_HISTOGRAM, _COUNT_COUNTER, _ERROR_COUNTER, _ACTIVE_UPDOWN  # noqa: PLW0603

    trace, metrics = _require_otel()

    # Use pixeltable.__version__ if it can be resolved; fall back to "" otherwise.
    try:
        import pixeltable

        version = getattr(pixeltable, '__version__', '') or ''
    except Exception:  # pragma: no cover - defensive
        version = ''

    _TRACER = trace.get_tracer('pixeltable', version)
    _METER = metrics.get_meter('pixeltable', version)

    _DURATION_HISTOGRAM = _METER.create_histogram(
        name='pixeltable.query.duration', unit='s', description='Wall-clock duration of a pixeltable operation.'
    )
    _COUNT_COUNTER = _METER.create_counter(
        name='pixeltable.query.count', unit='1', description='Number of pixeltable operations that completed.'
    )
    _ERROR_COUNTER = _METER.create_counter(
        name='pixeltable.query.errors', unit='1', description='Number of pixeltable operations that raised.'
    )
    _ACTIVE_UPDOWN = _METER.create_up_down_counter(
        name='pixeltable.query.active', unit='1', description='Number of currently-in-flight pixeltable operations.'
    )


def _safe_table_name(self_obj: Any) -> str | None:
    """Best-effort extraction of a pixeltable table name. Never raises."""
    for attr in ('_name', 'name', '_tbl_name'):
        try:
            v = getattr(self_obj, attr, None)
        except Exception:  # pragma: no cover - defensive
            v = None
        if isinstance(v, str) and v:
            return v
    # Fall back to first table on a Query
    try:
        first = getattr(self_obj, '_first_tbl', None)
        if first is not None:
            name = getattr(first, 'tbl_name', None)
            if callable(name):
                result = name()
                if isinstance(result, str):
                    return result
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def _query_first_table(self_obj: Any) -> str | None:
    """Pixeltable-specific custom attribute for Query spans."""
    try:
        first = getattr(self_obj, '_first_tbl', None)
        if first is None:
            return None
        name = getattr(first, 'tbl_name', None)
        if callable(name):
            result = name()
            if isinstance(result, str):
                return result
    except Exception:  # pragma: no cover - defensive
        pass
    return None


def _wrap_method(
    cls: type,
    method_name: str,
    span_name: str,
    operation: str,
    *,
    include_db_name: bool,
    extra_attrs: Callable[[Any, tuple, dict], dict[str, Any]] | None = None,
) -> None:
    """Replace ``cls.method_name`` with an OTel-instrumented version.

    Idempotent: if the method is already wrapped, this is a no-op.
    """
    original = cls.__dict__.get(method_name)
    if original is None:
        # Method not defined directly on the class (might be inherited). Pull
        # via getattr so we can still wrap it on this class.
        original = getattr(cls, method_name, None)
    if original is None:
        _LOG.debug('OTel: skipping %s.%s (not found)', cls.__name__, method_name)
        return
    if getattr(original, _INSTRUMENTED_MARKER, False):
        return

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # If instrumentation has been torn down between import and call,
        # behave as a transparent pass-through.
        tracer = _TRACER
        if tracer is None:
            return original(self, *args, **kwargs)

        attrs: dict[str, Any] = {_DB_SYSTEM_KEY: _DB_SYSTEM_VALUE, _DB_OPERATION_KEY: operation}
        if include_db_name:
            name = _safe_table_name(self)
            if name is not None:
                attrs[_DB_NAME_KEY] = name
        else:
            first = _query_first_table(self)
            if first is not None:
                attrs['pixeltable.first_table'] = first
        if extra_attrs is not None:
            try:
                attrs.update(extra_attrs(self, args, kwargs))
            except Exception:  # pragma: no cover - defensive
                pass

        metric_attrs = {_DB_SYSTEM_KEY: _DB_SYSTEM_VALUE, _DB_OPERATION_KEY: operation}

        # CLIENT span -- pixeltable operation is a database client call.
        # Import SpanKind lazily so module import doesn't require otel.
        from opentelemetry.trace import SpanKind, Status, StatusCode

        if _ACTIVE_UPDOWN is not None:
            _ACTIVE_UPDOWN.add(1, metric_attrs)
        start = time.perf_counter()
        try:
            with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT, attributes=attrs) as span:
                try:
                    result = original(self, *args, **kwargs)
                except BaseException as exc:
                    error_type = f'{type(exc).__module__}.{type(exc).__qualname__}'
                    if _ERROR_COUNTER is not None:
                        _ERROR_COUNTER.add(1, {**metric_attrs, 'error.type': error_type})
                    try:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    except Exception:  # pragma: no cover - defensive
                        pass
                    raise
                else:
                    # Best-effort row_count attribute for write ops returning UpdateStatus.
                    rc = _extract_row_count(result)
                    if rc is not None:
                        span.set_attribute('pixeltable.row_count', rc)
                    elif operation == _OP_SELECT:
                        try:
                            length = len(result)  # ResultSet supports len()
                            span.set_attribute('pixeltable.row_count', length)
                        except Exception:  # pragma: no cover - defensive
                            pass
                    return result
        finally:
            elapsed = time.perf_counter() - start
            if _DURATION_HISTOGRAM is not None:
                _DURATION_HISTOGRAM.record(elapsed, metric_attrs)
            if _COUNT_COUNTER is not None:
                _COUNT_COUNTER.add(1, metric_attrs)
            if _ACTIVE_UPDOWN is not None:
                _ACTIVE_UPDOWN.add(-1, metric_attrs)

    setattr(wrapper, _INSTRUMENTED_MARKER, True)
    # Also stash the original so tests / future tooling can inspect it.
    setattr(wrapper, '__pxt_otel_original__', original)  # noqa: B010
    setattr(cls, method_name, wrapper)


def _extract_row_count(result: Any) -> int | None:
    """Pull a row count off an UpdateStatus-like return value, if present."""
    if result is None:
        return None
    for attr in ('num_rows', 'num_inserted_rows', 'num_updated_rows', 'num_deleted_rows'):
        v = getattr(result, attr, None)
        if isinstance(v, int):
            return v
    return None


def _insert_attrs(self: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    on_error = kwargs.get('on_error')
    if on_error is None:
        return {}
    return {'pixeltable.on_error': str(on_error)}


def _update_attrs(self: Any, args: tuple, kwargs: dict) -> dict[str, Any]:
    cascade = kwargs.get('cascade')
    if cascade is None and len(args) >= 3:
        cascade = args[2]
    if cascade is None:
        return {}
    return {'pixeltable.cascade': bool(cascade)}


def instrument() -> None:
    """Wire OpenTelemetry instrumentation into pixeltable's core operations.

    This function is **idempotent**: calling it more than once has no
    additional effect. It must be called *after* ``import pixeltable`` but
    *before* the first pixeltable operation you want traced.

    The host application is responsible for configuring the
    :class:`opentelemetry.sdk.trace.TracerProvider` and
    :class:`opentelemetry.sdk.metrics.MeterProvider`. If neither is
    configured, the OpenTelemetry API hands back no-op implementations and
    this instrumentation emits nothing (and adds no measurable overhead).

    Raises:
        ImportError: If the optional ``opentelemetry-api`` / ``-sdk`` /
            ``-semantic-conventions`` packages are not installed. Install
            them with ``pip install 'pixeltable[otel]'``.
    """
    global _INSTRUMENTED  # noqa: PLW0603

    with _INSTRUMENT_LOCK:
        if _INSTRUMENTED:
            return

        _build_instruments()

        # Import the targets lazily so this module is import-safe before
        # pixeltable's catalog has been fully loaded.
        from pixeltable._query import Query
        from pixeltable.catalog.insertable_table import InsertableTable
        from pixeltable.catalog.table import Table

        _wrap_method(Query, 'collect', 'pixeltable.query.collect', _OP_SELECT, include_db_name=False)
        _wrap_method(Query, 'count', 'pixeltable.query.count', _OP_SELECT, include_db_name=False)

        _wrap_method(
            InsertableTable,
            'insert',
            'pixeltable.table.insert',
            _OP_INSERT,
            include_db_name=True,
            extra_attrs=_insert_attrs,
        )
        _wrap_method(InsertableTable, 'delete', 'pixeltable.table.delete', _OP_DELETE, include_db_name=True)
        _wrap_method(
            Table, 'update', 'pixeltable.table.update', _OP_UPDATE, include_db_name=True, extra_attrs=_update_attrs
        )
        _wrap_method(
            Table,
            'batch_update',
            'pixeltable.table.batch_update',
            _OP_UPDATE,
            include_db_name=True,
            extra_attrs=_update_attrs,
        )

        _INSTRUMENTED = True


def is_instrumented() -> bool:
    """Return ``True`` if :func:`instrument` has been called successfully."""
    return _INSTRUMENTED


__all__ = ['instrument', 'is_instrumented']
