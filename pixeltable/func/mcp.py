"""MCP (Model Context Protocol) integration -- importing MCP tools as Pixeltable UDFs.

This module provides the ``mcp_udfs`` function, which connects to an MCP server,
lists its available tools, and converts each tool into a Pixeltable CallableFunction.
This allows MCP tools to be used as computed columns, in queries, or as LLM tools
within Pixeltable workflows.

The conversion process:
1. Connects to the MCP server via the Streamable HTTP transport.
2. Calls ``session.list_tools()`` to enumerate available tools.
3. For each tool, converts its JSON Schema ``inputSchema`` to Pixeltable types
   and creates an async CallableFunction that invokes the tool via MCP.

All MCP tool functions:
- Accept keyword-only parameters (matching the tool's input schema).
- Return ``StringType`` (the text content of the tool's response).
- Are async (each invocation opens a new MCP session).

Dependencies:
    Requires the ``mcp`` package (checked at runtime via ``Env.require_package``).
"""

import inspect
from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable import exceptions as excs, type_system as ts
from pixeltable.env import Env
from pixeltable.func.signature import Parameter

if TYPE_CHECKING:
    import mcp


def mcp_udfs(url: str) -> list['pxt.func.Function']:
    """Connect to an MCP server and return its tools as Pixeltable functions.

    Synchronous wrapper around ``mcp_udfs_async``.

    Args:
        url: The HTTP URL of the MCP server.

    Returns:
        A list of CallableFunction instances, one per MCP tool.
    """
    from pixeltable.runtime import get_runtime

    return get_runtime().run_coro(mcp_udfs_async(url))


async def mcp_udfs_async(url: str) -> list['pxt.func.Function']:
    """Async implementation: connects to MCP server, lists tools, and converts each to a UDF."""
    Env.get().require_package('mcp')
    import mcp
    from mcp.client.streamable_http import streamablehttp_client

    list_tools_result: mcp.types.ListToolsResult | None = None
    async with (
        streamablehttp_client(url) as (read_stream, write_stream, _),
        mcp.ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        list_tools_result = await session.list_tools()
    assert list_tools_result is not None

    return [mcp_tool_to_udf(url, tool) for tool in list_tools_result.tools]


def mcp_tool_to_udf(url: str, mcp_tool: 'mcp.types.Tool') -> 'pxt.func.Function':
    """Convert a single MCP tool definition into a Pixeltable CallableFunction.

    Creates an async callable that opens an MCP session and invokes the tool,
    then wraps it in a CallableFunction with a signature derived from the
    tool's JSON Schema input schema.

    Args:
        url: The MCP server URL.
        mcp_tool: The MCP tool definition (name, description, inputSchema).

    Returns:
        A CallableFunction that invokes the MCP tool.
    """
    Env.get().require_package('mcp')
    import mcp
    from mcp.client.streamable_http import streamablehttp_client

    async def invoke(**kwargs: Any) -> str:
        # TODO: Cache session objects rather than creating a new one each time?
        async with (
            streamablehttp_client(url) as (read_stream, write_stream, _),
            mcp.ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            res = await session.call_tool(name=mcp_tool.name, arguments=kwargs)
            # TODO Handle image/audio responses?
            return res.content[0].text  # type: ignore[union-attr]

    if mcp_tool.description is not None:
        invoke.__doc__ = mcp_tool.description

    input_schema = mcp_tool.inputSchema
    params = {
        name: __mcp_param_to_pxt_type(mcp_tool.name, name, param) for name, param in input_schema['properties'].items()
    }
    required = input_schema.get('required', [])

    # Ensure that any params not appearing in `required` are nullable.
    # (A required param might or might not be nullable, since its type might be an 'anyOf' containing a null.)
    for name in params.keys() - required:
        params[name] = params[name].copy(nullable=True)

    signature = pxt.func.Signature(
        return_type=ts.StringType(),  # Return type is always string
        parameters=[Parameter(name, col_type, inspect.Parameter.KEYWORD_ONLY) for name, col_type in params.items()],
    )

    return pxt.func.CallableFunction(signatures=[signature], py_fns=[invoke], self_name=mcp_tool.name)


def __mcp_param_to_pxt_type(tool_name: str, name: str, param: dict[str, Any]) -> ts.ColumnType:
    """Convert an MCP tool parameter's JSON Schema definition to a Pixeltable ColumnType.

    Args:
        tool_name: Name of the MCP tool (for error messages).
        name: Name of the parameter (for error messages).
        param: The JSON Schema definition of the parameter.

    Returns:
        The corresponding Pixeltable ColumnType.

    Raises:
        excs.Error: If the JSON Schema type cannot be mapped to a Pixeltable type.
    """
    pxt_type = ts.ColumnType.from_json_schema(param)
    if pxt_type is None:
        raise excs.Error(f'Unknown type schema for MCP parameter {name!r} of tool {tool_name!r}: {param}')
    return pxt_type
