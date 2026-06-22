import logging
import socket
import subprocess
import sys
import time
from typing import Iterator

import pytest

import pixeltable as pxt

from .utils import rerun, skip_test_if_no_client, skip_test_if_not_installed

_logger = logging.getLogger('pixeltable')

MCP_SERVER_URL = 'http://127.0.0.1:8000/mcp'


@rerun(reruns=3, delay=30)
class TestMcp:
    def test_mcp_server(self, uses_db: None, init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp')

        udfs = pxt.mcp_udfs(MCP_SERVER_URL)
        assert udfs[0].name == 'pixelmultiple'
        assert udfs[0].comment() == 'Computes the Pixelmultiple of two integers.'
        assert udfs[1].name == 'pixeldict'
        assert udfs[1].comment() == 'Returns the Pixeldict of a dictionary.'

        t = pxt.create_table('test_mcp', {'a': pxt.Int, 'b': pxt.Int})
        t.add_computed_column(pixelmultiple=udfs[0](a=t.a, b=t.b))
        t.insert([{'a': 3, 'b': 4}, {'a': 5, 'b': 6}])
        res = t.order_by(t.a).collect()
        assert res[0]['pixelmultiple'] == str((3 + 22) * 4)
        assert res[1]['pixelmultiple'] == str((5 + 22) * 6)

    def test_mcp_as_tools(self, uses_db: None, init_mcp_server: None) -> None:
        skip_test_if_not_installed('mcp', 'openai')
        skip_test_if_no_client('openai')
        from pixeltable.functions import openai

        udfs = pxt.mcp_udfs(MCP_SERVER_URL)
        tools = pxt.tools(*udfs)

        t = pxt.create_table('test_mcp', {'prompt': pxt.String})
        messages = [{'role': 'user', 'content': t.prompt}]
        t.add_computed_column(response=openai.chat_completions(messages, model='gpt-4o-mini', tools=tools))
        t.add_computed_column(tool_calls=openai.invoke_tools(tools, t.response))
        t.insert(prompt='What is the pixelmultiple of 7 and 9?')
        res = t.head()
        assert res[0]['tool_calls'] == {'pixelmultiple': [str((7 + 22) * 9)], 'pixeldict': None}


def _wait_for_server(host: str, port: int, timeout: float, process: subprocess.Popen[bytes]) -> None:
    """Poll until the server accepts TCP connections or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'MCP server process exited with code {process.returncode}')
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f'MCP server on {host}:{port} did not start within {timeout}s')


@pytest.fixture(scope='session')
def init_mcp_server(init_env: None) -> Iterator[None]:
    skip_test_if_not_installed('mcp')

    _logger.info('Starting MCP server pytest fixture.')
    mcp_process = subprocess.Popen([sys.executable, 'tests/example_mcp_server.py'])
    _wait_for_server('127.0.0.1', 8000, timeout=30, process=mcp_process)
    yield

    _logger.info('Terminating MCP server pytest fixture.')
    mcp_process.kill()
