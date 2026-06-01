"""Optional OpenTelemetry instrumentation for Pixeltable.

This package is a no-op unless the optional `otel` extra is installed and the
host application calls :func:`pixeltable.telemetry.otel.instrument`. See
``docs/release/opentelemetry.md`` for full documentation.

Importing this package never imports the OpenTelemetry SDK eagerly; that
import only happens inside :func:`pixeltable.telemetry.otel.instrument` so
that ``import pixeltable`` keeps working when the SDK is not installed.
"""

from . import otel

__all__ = ['otel']
