# pixeltable.func -- Function Infrastructure

This package provides the core infrastructure for all function types in Pixeltable:
user-defined functions (UDFs), aggregate functions (UDAs), query templates, expression
templates, iterators, and LLM tool integration.

## Module Overview

The `func/` package is the backbone of Pixeltable's declarative computation model.
When users define computed columns, add embedding indexes, or call AI integrations,
they are working with objects from this package. Functions are first-class citizens
in Pixeltable -- they carry typed signatures, support polymorphic overloading,
can be serialized for persistence, and integrate with the query execution engine.

## Function Type Hierarchy

```
Function (ABC)                         # Base class for all function types
|
+-- CallableFunction                   # Python callable backed UDF (@pxt.udf)
|   - Wraps a Python function/lambda/coroutine
|   - Supports batch execution (batch_size parameter)
|   - Two serialization modes: by module path or by cloudpickle (stored)
|   - Supports overloading via .overload() decorator
|
+-- AggregateFunction                  # Aggregation via Aggregator subclass (@pxt.uda)
|   - Wraps an Aggregator class (init/update/value protocol)
|   - Supports order_by and group_by for windowed aggregation
|   - Signature inferred from update() and __init__() methods
|
+-- ExprTemplateFunction               # Parameterized expression template
|   - Created by Function.using(), @pxt.expr_udf, or pxt.udf(table)
|   - Wraps a Pixeltable expression tree with Variable placeholders
|   - Substitutes arguments into the template at call time
|
+-- QueryTemplateFunction              # Parameterized query template (@pxt.query)
|   - Wraps a parameterized Query (DataFrame-like) object
|   - Always async; returns JSON (list of dicts)
|   - Used for retrieval UDFs and tool integration
|
+-- InvalidFunction                    # Placeholder for unresolvable functions
    - Preserves original metadata for re-serialization
    - Defers errors to invocation time


GeneratingFunction                     # Iterator-based table-generating function (@pxt.iterator)
|   - Wraps a PxtIterator class or generator function
|   - Produces multiple output rows per input row (one-to-many)
|   - Output schema inferred from TypedDict or conditional_output_schema
|
+-- InvalidGeneratingFunction          # Placeholder for unresolvable iterators


Aggregator (ABC)                       # Base class for user-defined aggregators
    - __init__(): Initialize state
    - update(): Process each row
    - value(): Return final result


Signature                              # Typed function signature
    - return_type, parameters, is_batched
    - Type inference from Python annotations
    - Argument validation and binding

Parameter                              # Single typed parameter
    - name, col_type, kind, default, is_batched


Tool / Tools / ToolChoice             # LLM tool integration (Pydantic models)
    - Wrap Functions for LLM function calling APIs
    - Serialize to JSON Schema format
    - Dispatch tool call results back to functions
```

## How @pxt.udf Works

The `@pxt.udf` decorator converts a Python function into a `CallableFunction`:

1. **Signature inference** (`Signature.create`): Inspects the function's type
   annotations to determine parameter types and return type. Handles `Batch[T]`
   annotations for batched parameters.

2. **Path determination**: If the function is defined in an importable module,
   its fully-qualified path (e.g. `mymodule.my_fn`) is used for serialization.
   If defined in a notebook or lambda, it will be stored via cloudpickle.

3. **Registration** (`FunctionRegistry.register_function`): Module-level functions
   are registered globally by their path. Functions with `is_method=True` or
   `is_property=True` are also indexed by their base type for method-style dispatch.

4. **CallableFunction creation**: The function, signature(s), and configuration
   are packaged into a `CallableFunction` instance.

### Decorator forms:

```python
# Without arguments
@pxt.udf
def my_fn(x: int) -> int:
    return x + 1

# With arguments
@pxt.udf(batch_size=32)
def my_batch_fn(x: Batch[int]) -> Batch[int]:
    return [v + 1 for v in x]

# From a table
tool_fn = pxt.udf(my_table, description='Lookup tool')
```

## How @pxt.uda Works

The `@pxt.uda` decorator converts an `Aggregator` subclass into an `AggregateFunction`:

1. **Signature inference**: Inspects `__init__()`, `update()`, and `value()` methods.
   `update()` parameters become positional arguments; `__init__()` parameters become
   keyword-only arguments. Return type comes from `value()`.

2. **Special parameters**: `order_by` and `group_by` are intercepted in `__call__()`
   and handled as windowing clauses rather than passed to the aggregator.

```python
@pxt.uda
class MyAvg(pxt.Aggregator):
    def __init__(self):
        self.total = 0.0
        self.count = 0
    def update(self, val: float) -> None:
        self.total += val
        self.count += 1
    def value(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
```

## How Functions Are Registered and Looked Up

### Registration (at module import time)

When a `@pxt.udf` decorator runs in a module:
1. `make_function()` in `udf.py` constructs the `CallableFunction`.
2. `validate_symbol_path()` in `globals.py` checks the path is well-formed.
3. `FunctionRegistry.register_function()` stores it in `module_fns[fqn]`.
4. For type methods/properties, it's also stored in `type_methods[base_type][name]`.

### Lookup (at deserialization time)

When loading a stored computed column:
1. `Function.from_dict()` reads the `_classpath` to determine the Function subclass.
2. For module functions: `resolve_symbol()` in `globals.py` re-imports the module
   and retrieves the function by attribute access.
3. For stored functions: `FunctionRegistry.get_stored_function()` loads from the
   database, deserializing via cloudpickle.
4. On failure: An `InvalidFunction` is returned, deferring the error.

## File-by-File Description

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports; re-exports all public classes and decorators |
| `function.py` | `Function` ABC and `InvalidFunction`; base class for all function types |
| `callable_function.py` | `CallableFunction`; wraps Python callables with batch/async support |
| `aggregate_function.py` | `AggregateFunction`, `Aggregator`, and `@pxt.uda` decorator |
| `expr_template_function.py` | `ExprTemplateFunction` and `ExprTemplate`; parameterized expressions |
| `query_template_function.py` | `QueryTemplateFunction`, `@pxt.query`, and `retrieval_udf` |
| `iterator.py` | `GeneratingFunction`, `PxtIterator`, `@pxt.iterator`; table-generating functions |
| `signature.py` | `Signature` and `Parameter`; type inference and argument validation |
| `function_registry.py` | `FunctionRegistry` singleton; global function catalog and DB storage |
| `globals.py` | Utility helpers: `resolve_symbol`, `validate_symbol_path`, `get_caller_module_path` |
| `udf.py` | `@pxt.udf`, `@pxt.expr_udf`, `make_function`, `from_table` |
| `tools.py` | `Tool`, `Tools`, `ToolChoice`; LLM tool calling integration |
| `mcp.py` | `mcp_udfs`; MCP server tool import |

## MCP and Tools Integration

### LLM Tools (`tools.py`)

The `Tool` and `Tools` classes wrap Pixeltable functions for LLM function calling:

- **Serialization**: `Tool` uses Pydantic's `model_serializer` to produce JSON Schema
  compatible with OpenAI/Anthropic APIs. Parameter types are converted via
  `ColumnType._to_json_schema()`.
- **Invocation**: When an LLM returns tool call results, `Tool.invoke()` creates
  expressions that extract typed arguments and call the underlying function.
- **Type extraction**: Private helper UDFs (`_extract_str_tool_arg`, etc.) handle
  converting JSON values to Pixeltable-typed values.

### MCP Integration (`mcp.py`)

The `mcp_udfs()` function connects to an MCP (Model Context Protocol) server and
imports its tools as Pixeltable functions:

1. Opens an MCP session via Streamable HTTP transport.
2. Lists available tools via `session.list_tools()`.
3. Converts each tool's JSON Schema input to Pixeltable types.
4. Creates async `CallableFunction` instances that invoke the MCP tools.

## How Function Signatures and Type Checking Work

### Type Inference (`Signature.create`)

1. Each parameter's Python type annotation is mapped to a `ColumnType` via
   `ColumnType.from_python_type()`.
2. `Batch[T]` annotations (implemented as `Annotated[list[T], 'pxt-batch']`) are
   detected and unwrapped; the parameter is marked as batched.
3. System parameters (e.g. `_runtime_ctx`) are filtered out.
4. Default values are converted to `Literal` expressions.

### Argument Validation (`Signature.validate_args`)

At call time, each argument's `col_type` is checked against the parameter's `col_type`:
- `is_supertype_of()` is used (with `ignore_nullable=True`, since Nones are handled
  by FunctionCall's null-skipping logic).
- JSON columns are allowed to match scalar parameter types (a practical concession).

### Overload Resolution (`Function._bind_to_matching_signature`)

For polymorphic functions (multiple signatures):
1. Try each signature in declaration order.
2. Use `inspect.Signature.bind()` for Python-level argument matching.
3. Then validate Pixeltable types via `validate_args()`.
4. Return the first successful match, or raise an error if none match.

### Conditional Return Types (`Function.conditional_return_type`)

Some functions have return types that depend on argument values (e.g. a function that
returns different types based on a `model` parameter). The `@fn.conditional_return_type`
decorator specifies a callable that receives constant argument values and returns the
specific return type for that call.

### Nullable Propagation (`Function.call_return_type`)

If a function's return type is non-nullable but it receives a nullable argument for a
non-nullable parameter, the return type is automatically made nullable. This is because
Pixeltable skips function evaluation when any non-nullable parameter is None.
