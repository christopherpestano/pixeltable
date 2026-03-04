"""pixeltable.func -- Infrastructure for Pixeltable's function system.

This package provides the core abstractions for user-defined functions (UDFs),
aggregate functions (UDAs), query templates, expression templates, iterators,
and tool integration within Pixeltable. It handles:

- Function declaration via decorators (@pxt.udf, @pxt.uda, @pxt.query, @pxt.iterator)
- Signature inference from Python type hints
- Function registration and lookup (FunctionRegistry)
- Serialization/deserialization of functions for persistence
- Polymorphic dispatch (overloaded signatures)
- Batch execution for vectorized operations
- LLM tool integration (Tool/Tools classes)
- MCP (Model Context Protocol) server integration

Class hierarchy::

    Function (ABC)
    +-- CallableFunction        # Python callable backed UDF
    +-- AggregateFunction       # Aggregation via Aggregator subclass
    +-- ExprTemplateFunction    # Parameterized expression template
    +-- QueryTemplateFunction   # Parameterized query template
    +-- InvalidFunction         # Placeholder for unresolvable functions

    GeneratingFunction          # Iterator-based table-generating function
    +-- InvalidGeneratingFunction

    Aggregator (ABC)            # Base class for user-defined aggregators

    Signature                   # Typed function signature
    Parameter                   # Individual typed parameter

    Tool / Tools / ToolChoice   # LLM tool integration wrappers
"""

# ruff: noqa: F401

from .aggregate_function import AggregateFunction, Aggregator, uda
from .callable_function import CallableFunction
from .expr_template_function import ExprTemplateFunction
from .function import Function, InvalidFunction
from .function_registry import FunctionRegistry
from .iterator import GeneratingFunction, GeneratingFunctionCall, PxtIterator, iterator
from .mcp import mcp_udfs
from .query_template_function import QueryTemplateFunction, query, retrieval_udf
from .signature import Batch, Parameter, Signature
from .tools import Tool, ToolChoice, Tools
from .udf import expr_udf, make_function, udf
