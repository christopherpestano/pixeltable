"""Tool and Tools -- LLM tool integration for Pixeltable functions.

This module provides the ``Tool``, ``Tools``, and ``ToolChoice`` classes that wrap
Pixeltable functions for use as LLM tools (function calling). These classes are
Pydantic models that can be serialized to JSON in the format expected by LLM APIs
(OpenAI, Anthropic, etc.).

Architecture:
    - ``Tool``: Wraps a single Pixeltable Function with optional name/description
      overrides. Serializes to JSON Schema format via ``model_serializer``.
      Provides ``invoke()`` to execute the tool from LLM tool-call output.
    - ``Tools``: A collection of Tool instances. Serializes to a list of tool
      definitions. Provides ``_invoke()`` to dispatch tool calls to the correct tool.
    - ``ToolChoice``: Configures how the LLM should select tools (auto, required,
      or a specific tool name).

The module also defines private helper UDFs (``_extract_str_tool_arg``, etc.) that
extract typed arguments from the generic dict format of LLM tool call responses.
These UDFs handle the type conversion from JSON values to Pixeltable-typed values.

Usage pattern:
    1. Create tools: ``tools = pxt.tools(my_fn1, my_fn2)``
    2. Pass to LLM: ``t.add_computed_column(response=openai.chat_completions(tools=tools))``
    3. Invoke results: ``t.add_computed_column(results=tools._invoke(t.response.tool_calls))``
"""

import json
import uuid
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import pydantic

from pixeltable import exceptions as excs, type_system as ts

from .function import Function
from .signature import Parameter
from .udf import udf

if TYPE_CHECKING:
    from pixeltable import exprs


# The Tool and Tools classes are containers that hold Pixeltable UDFs and related metadata, so that they can be
# realized as LLM tools. They are implemented as Pydantic models in order to provide a canonical way of converting
# to JSON, via the Pydantic `model_serializer` interface. In this way, they can be passed directly as UDF
# parameters as described in the `pixeltable.tools` and `pixeltable.tool` docstrings.
#
# (The dataclass dict serializer is insufficiently flexible for this purpose: `Tool` contains a member of type
# `Function`, which is not natively JSON-serializable; Pydantic provides a way of customizing its default
# serialization behavior, whereas dataclasses do not.)


class Tool(pydantic.BaseModel):
    """Wraps a Pixeltable Function for use as an LLM tool.

    Serializes to JSON Schema format compatible with OpenAI/Anthropic tool calling APIs.
    The ``model_serializer`` produces a dict with 'name', 'description', 'parameters'
    (JSON Schema object), and 'required' fields.

    Attributes:
        fn: The Pixeltable Function to expose as a tool.
        name: Optional override for the tool name (defaults to fn.name).
        description: Optional override for the tool description (defaults to fn.comment()).
    """

    # Allow arbitrary types so that we can include a Pixeltable function in the schema.
    # We will implement a model_serializer to ensure the Tool model can be serialized.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    fn: Function
    name: str | None = None
    description: str | None = None

    @property
    def parameters(self) -> dict[str, Parameter]:
        return self.fn.signature.parameters

    @pydantic.model_serializer
    def ser_model(self) -> dict[str, Any]:
        return {
            'name': self.name or self.fn.name,
            'description': self.description or self.fn.comment(),
            'parameters': {
                'type': 'object',
                'properties': {param.name: param.col_type._to_json_schema() for param in self.parameters.values()},
            },
            'required': [param.name for param in self.parameters.values() if not param.col_type.nullable],
            'additionalProperties': False,  # TODO Handle kwargs?
        }

    def invoke(self, tool_calls: 'exprs.Expr') -> 'exprs.Expr':
        """Create an expression that executes this tool from LLM tool-call output.

        The tool_calls expression must be in standardized format:
        ``{tool_name: [{'args': {name1: value1, ...}}, ...], ...}``

        Returns an expression that maps over the tool's invocations and calls
        the underlying function with the extracted arguments.
        """
        import pixeltable.functions as pxtf

        func_name = self.name or self.fn.name
        return pxtf.map(tool_calls[func_name]['*'], lambda x: self.__invoke_kwargs(x.args))

    def __invoke_kwargs(self, kwargs: 'exprs.Expr') -> 'exprs.FunctionCall':
        kwargs = {param.name: self.__extract_tool_arg(param, kwargs) for param in self.parameters.values()}
        return self.fn(**kwargs)

    def __extract_tool_arg(self, param: Parameter, kwargs: 'exprs.Expr') -> 'exprs.FunctionCall':
        if param.col_type.is_string_type():
            return _extract_str_tool_arg(kwargs, param_name=param.name)
        if param.col_type.is_int_type():
            return _extract_int_tool_arg(kwargs, param_name=param.name)
        if param.col_type.is_float_type():
            return _extract_float_tool_arg(kwargs, param_name=param.name)
        if param.col_type.is_bool_type():
            return _extract_bool_tool_arg(kwargs, param_name=param.name)
        if param.col_type.is_json_type():
            return _extract_json_tool_arg(kwargs, param_name=param.name)
        if param.col_type.is_uuid_type():
            return _extract_uuid_tool_arg(kwargs, param_name=param.name)
        raise AssertionError(param.col_type)


class ToolChoice(pydantic.BaseModel):
    """Configuration for how an LLM should select tools.

    Exactly one of ``auto``, ``required``, or ``tool`` must be set.

    Attributes:
        auto: If True, the LLM decides whether to use tools.
        required: If True, the LLM must use at least one tool.
        tool: If set, the LLM must use the specified tool by name.
        parallel_tool_calls: If True, allow the LLM to call multiple tools in parallel.
    """

    auto: bool
    required: bool
    tool: str | None
    parallel_tool_calls: bool


class Tools(pydantic.BaseModel):
    """A collection of Tool instances for LLM tool calling.

    Serializes to a list of tool definitions in JSON Schema format.
    Provides ``_invoke()`` to dispatch LLM tool call results to the correct tools.

    Attributes:
        tools: The list of Tool wrappers.
    """

    tools: list[Tool]

    @pydantic.model_serializer
    def ser_model(self) -> list[dict[str, Any]]:
        return [tool.ser_model() for tool in self.tools]

    # `tool_calls` must be in standardized tool invocation format:
    # {tool_name: {'args': {name1: value1, name2: value2, ...}}, ...}
    def _invoke(self, tool_calls: 'exprs.Expr') -> 'exprs.InlineDict':
        from pixeltable import exprs

        return exprs.InlineDict({tool.name or tool.fn.name: tool.invoke(tool_calls) for tool in self.tools})

    def choice(
        self,
        auto: bool = False,
        required: bool = False,
        tool: str | Function | None = None,
        parallel_tool_calls: bool = True,
    ) -> ToolChoice:
        """Create a ToolChoice configuration for these tools.

        Exactly one of ``auto``, ``required``, or ``tool`` must be specified.

        Args:
            auto: Let the LLM decide whether to call tools.
            required: Force the LLM to call at least one tool.
            tool: Force the LLM to call a specific tool (by name or Function reference).
            parallel_tool_calls: Allow parallel tool invocations.

        Returns:
            A ToolChoice configuration object.

        Raises:
            excs.Error: If not exactly one option is specified, or if the named tool
                is not in the tools list.
        """
        if sum([auto, required, tool is not None]) != 1:
            raise excs.Error('Exactly one of `auto`, `required`, or `tool` must be specified.')
        tool_name: str | None = None
        if tool is not None:
            try:
                tool_obj = next(
                    t
                    for t in self.tools
                    if (isinstance(tool, Function) and t.fn == tool)
                    or (isinstance(tool, str) and (t.name or t.fn.name) == tool)
                )
                tool_name = tool_obj.name or tool_obj.fn.name
            except StopIteration:
                raise excs.Error(f'That tool is not in the specified list of tools: {tool}') from None
        return ToolChoice(auto=auto, required=required, tool=tool_name, parallel_tool_calls=parallel_tool_calls)


@udf
def _extract_str_tool_arg(kwargs: dict[str, Any], param_name: str) -> str | None:
    return _extract_arg(str, kwargs, param_name)


@udf
def _extract_int_tool_arg(kwargs: dict[str, Any], param_name: str) -> int | None:
    return _extract_arg(int, kwargs, param_name)


@udf
def _extract_float_tool_arg(kwargs: dict[str, Any], param_name: str) -> float | None:
    return _extract_arg(float, kwargs, param_name)


@udf
def _extract_bool_tool_arg(kwargs: dict[str, Any], param_name: str) -> bool | None:
    return _extract_arg(bool, kwargs, param_name)


@udf
def _extract_json_tool_arg(kwargs: dict[str, Any], param_name: str) -> ts.Json | None:
    if param_name in kwargs:
        return json.loads(kwargs[param_name])
    return None


@udf
def _extract_uuid_tool_arg(kwargs: dict[str, Any], param_name: str) -> uuid.UUID | None:
    return _extract_arg(uuid.UUID, kwargs, param_name)


T = TypeVar('T')


def _extract_arg(eval_fn: Callable[[Any], T], kwargs: dict[str, Any], param_name: str) -> T | None:
    if param_name in kwargs:
        return eval_fn(kwargs[param_name])
    return None
