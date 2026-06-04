# OpenTelemetry Instrumentation Spec for Pixeltable

This document specifies the OpenTelemetry **spans**, **metrics**, and **resource
attributes** emitted by the pixeltable instrumentation (module
`pixeltable.telemetry.otel`). It is the contract that the implementation in
KAN-32 follows.

The spec deliberately stays close to the OTel semantic conventions for
databases so that operators familiar with SQLAlchemy or Snowflake observability
stacks can immediately read pixeltable telemetry.

## Reference instrumentations surveyed

The taxonomy below is derived from these public references:

- **OpenTelemetry semantic conventions — database** (`db.*` attributes):
  <https://opentelemetry.io/docs/specs/semconv/database/database-spans/>
  and <https://opentelemetry.io/docs/specs/semconv/database/database-metrics/>.
  Defines `db.system`, `db.operation` / `db.operation.name`, `db.name`,
  `db.statement`, and span kind expectations for client database calls.
- **`opentelemetry-instrumentation-sqlalchemy`**:
  <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/sqlalchemy/sqlalchemy.html>
  and <https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation/opentelemetry-instrumentation-sqlalchemy>.
  Reference implementation for emitting CLIENT spans around DB statements and
  for the duration-of-query pattern in the Python ecosystem.
- **Snowflake OpenTelemetry / observability documentation**:
  <https://docs.snowflake.com/en/user-guide/logging/logging-tracing-getting-started>
  and <https://docs.snowflake.com/en/developer-guide/logging-tracing/tracing-python>.
  Snowflake emits spans tagged with `db.system="snowflake"`, attaches
  `db.statement` where appropriate, and uses `service.name` / `service.version`
  resource attributes. Pixeltable mirrors that resource shape.

## Resource attributes

The instrumentation does **not** install a `Resource` of its own (the host app
owns the `TracerProvider`/`MeterProvider`). However, the helper that builds the
default `TracerProvider` in the validation script (`scripts/validate_otel.py`)
attaches:

| Resource attribute | Value                                  | Notes                                   |
|--------------------|----------------------------------------|-----------------------------------------|
| `service.name`     | `"pixeltable"`                         | Stable identifier for the service.      |
| `service.version`  | `pixeltable.__version__`               | Tracks the installed library version.   |
| `db.system`        | `"pixeltable"`                         | Mirrored as both span attr and resource.|

When the host application already configures its own `Resource`, the
instrumentation does not override it — `instrument()` only uses
`trace.get_tracer("pixeltable", pixeltable.__version__)` and
`metrics.get_meter("pixeltable", pixeltable.__version__)`, which inherits the
ambient resource. The two attributes above are the resource keys the user is
expected to set on their own provider for pixeltable to be identifiable.

## Spans

All spans are emitted with kind **CLIENT** (we treat the pixeltable
operation as a database client call, matching the `db.system` semconv).

| Span name                       | Kind   | Pixeltable entry-point                                  | Required attributes                                        |
|---------------------------------|--------|---------------------------------------------------------|------------------------------------------------------------|
| `pixeltable.query.collect`      | CLIENT | `pixeltable._query.Query.collect`                       | `db.system`, `db.operation`, `db.name?`                    |
| `pixeltable.query.count`        | CLIENT | `pixeltable._query.Query.count`                         | `db.system`, `db.operation`, `db.name?`                    |
| `pixeltable.table.insert`       | CLIENT | `pixeltable.catalog.InsertableTable.insert`             | `db.system`, `db.operation`, `db.name`                     |
| `pixeltable.table.update`       | CLIENT | `pixeltable.catalog.Table.update`                       | `db.system`, `db.operation`, `db.name`                     |
| `pixeltable.table.batch_update` | CLIENT | `pixeltable.catalog.Table.batch_update`                 | `db.system`, `db.operation`, `db.name`                     |
| `pixeltable.table.delete`       | CLIENT | `pixeltable.catalog.InsertableTable.delete`             | `db.system`, `db.operation`, `db.name`                     |

`db.name` is the pixeltable table name (`Table._name`) where available. It is
omitted on `Query` spans because a `Query` may join multiple tables; the first
table name, if any, is recorded under the custom attribute
`pixeltable.first_table` (see "Custom attributes" below).

On failure, every span sets its status to `ERROR` and records the exception via
`span.record_exception(exc)`, per the OTel spec for exception events.

### Standard `db.*` attribute values

| Attribute       | Value                                                                 |
|-----------------|-----------------------------------------------------------------------|
| `db.system`     | `"pixeltable"` on every span.                                         |
| `db.operation`  | `"select"` (collect/count), `"insert"`, `"update"`, `"delete"`.       |
| `db.name`       | The pixeltable table name when the span is bound to a single table.   |

### Custom attributes

These are pixeltable-specific keys, prefixed `pixeltable.*`, used where no
standard semconv key applies. Each entry below carries a short rationale.

| Custom attribute            | Used on                       | Rationale                                                                                                                      |
|-----------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `pixeltable.first_table`    | `pixeltable.query.collect`, `pixeltable.query.count` | A `Query` can join multiple tables; we surface the first one for grep/filter convenience without claiming it's the only table. |
| `pixeltable.row_count`      | `pixeltable.query.collect`, all write spans (when known) | Number of rows returned/inserted/updated/deleted. Maps to `db.response.returned_rows` in newer semconv but is renamed under the `pixeltable.*` namespace to keep the spec stable across semconv revisions. |
| `pixeltable.cascade`        | `pixeltable.table.update`     | Whether the update cascaded to computed columns; pixeltable-specific.                                                          |
| `pixeltable.on_error`       | `pixeltable.table.insert`     | The `on_error` mode (`abort`/`ignore`); pixeltable-specific.                                                                   |

## Metrics

All metrics live under the `pixeltable.*` namespace. The instrument names
mirror the spans, so an operator can group a histogram and counter by the same
`db.operation` value.

| Instrument name             | Type            | Unit | Description                                            | Attribute keys                |
|-----------------------------|-----------------|------|--------------------------------------------------------|-------------------------------|
| `pixeltable.query.duration` | Histogram       | `s`  | Wall-clock duration of a pixeltable operation.         | `db.system`, `db.operation`   |
| `pixeltable.query.count`    | Counter         | `1`  | Number of pixeltable operations that completed.        | `db.system`, `db.operation`   |
| `pixeltable.query.errors`   | Counter         | `1`  | Number of pixeltable operations that raised.           | `db.system`, `db.operation`, `error.type` |
| `pixeltable.query.active`   | UpDownCounter   | `1`  | Number of currently-in-flight pixeltable operations.   | `db.system`, `db.operation`   |

`pixeltable.query.active` is the analogue of the "active operations gauge" the
KAN-31 spec template called for. We use an `UpDownCounter` rather than an
observable gauge so the implementation can increment on entry / decrement on
exit synchronously and we don't need to manage a callback.

### Unit choice

`pixeltable.query.duration` is recorded in **seconds** to match the OTel
guidance for histogram units (`s` is preferred over `ms` in the database
semconv). All counts use `"1"`, also per semconv.

### Attribute values

- `db.system` is always `"pixeltable"`.
- `db.operation` takes the same values as the corresponding span:
  `select`, `insert`, `update`, `delete`.
- `error.type` (on `pixeltable.query.errors`) is the fully qualified exception
  class name (e.g. `pixeltable.exceptions.Error`).

## No-op behavior

When the host application has not configured a `TracerProvider`/`MeterProvider`,
`opentelemetry.trace.get_tracer(...)` returns a no-op tracer and
`opentelemetry.metrics.get_meter(...)` returns a no-op meter. The
instrumentation therefore adds **zero** observable overhead and emits no
spans/metrics unless the host opts in. This is verified by a dedicated unit
test in `tests/test_otel.py`.

## Idempotency

`pixeltable.telemetry.otel.instrument()` is idempotent: calling it multiple
times wraps the entry-points exactly once. The implementation tags each
wrapped function with `__pxt_otel_instrumented__ = True` and short-circuits if
the tag is already present.
