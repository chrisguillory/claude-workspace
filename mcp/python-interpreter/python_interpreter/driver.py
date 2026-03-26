"""
External interpreter driver script.

This script is injected into external interpreter subprocesses to provide
a communication bridge with the MCP server. It speaks a length-prefixed
JSON protocol over stdin/stdout.

Protocol:
    Request:  "{length}\n{json_request}"
    Response: "{length}\n{json_response}"

Request format:
    {"action": "execute", "code": "..."}
    {"action": "shutdown"}

Response format:
    {"stdout": "...", "stderr": "...", "result": "...", "error": null}
    {"error": "ErrorType: message", "traceback": "..."}

This script is designed to be completely standalone - it cannot import from
the MCP server since it runs in a different Python environment.
"""

from __future__ import annotations

__all__ = [
    'DriverRequest',
    'ErrorResponse',
    'ExecuteResponse',
    'ReadyResponse',
    'ShutdownResponse',
    'execute_code',
    'main',
    'read_request',
    'send_response',
]

import ast
import contextlib
import io
import json
import sys
import traceback
from typing import Any, Literal, NotRequired, TypedDict, cast

# -- Protocol types (self-contained — driver runs in external interpreter env) -


class DriverRequest(TypedDict):
    """Length-prefixed JSON request from MCP server."""

    action: str
    code: NotRequired[str]


class ExecuteResponse(TypedDict):
    """Response from execute action. Validated by DriverExecuteResponse on consumer side."""

    stdout: str
    stderr: str
    result: str
    error: str | None
    error_type: str | None
    module_name: str | None


class ReadyResponse(TypedDict):
    """Startup signal. Validated by DriverReadyResponse on consumer side."""

    status: Literal['ready']
    python_version: str
    python_executable: str


class ShutdownResponse(TypedDict):
    """Clean shutdown acknowledgment."""

    status: str


class ErrorResponse(TypedDict):
    """Protocol or unknown-action error."""

    error: str
    traceback: None


type DriverMessage = ExecuteResponse | ReadyResponse | ShutdownResponse | ErrorResponse


# Persistent scope for this interpreter - survives across requests
_scope_globals: dict[str, Any] = {}  # strict_typing_linter.py: mutable-type — exec/eval scope mutated by Python runtime


def read_request() -> DriverRequest | None:
    """Read a length-prefixed JSON request from stdin.

    Returns:
        Parsed request, or None if stdin is closed/EOF.
    """
    try:
        length_line = sys.stdin.readline()
        if not length_line:
            return None  # EOF

        length = int(length_line.strip())
        if length <= 0:
            return None

        json_data = sys.stdin.read(length)
        if len(json_data) < length:
            return None  # Incomplete read

        return cast(DriverRequest, json.loads(json_data))  # static-only; no runtime validation

    except (ValueError, json.JSONDecodeError) as e:
        send_response(
            ExecuteResponse(
                stdout='',
                stderr='',
                result='',
                error=f'ProtocolError: {e}',
                error_type=None,
                module_name=None,
            ),
        )
        return None


def send_response(response: DriverMessage) -> None:
    """Send a length-prefixed JSON response to stdout."""
    json_data = json.dumps(response)
    sys.stdout.write(f'{len(json_data)}\n{json_data}')
    sys.stdout.flush()


def _split_last_expression(code: str) -> tuple[str, str] | None:
    """Split code into preceding statements and final expression.

    Uses ast.unparse on AST nodes for splitting, which correctly handles
    semicolons (multiple statements on one line) and multi-line expressions.

    Returns:
        (preceding_code, expression_code) or None if last statement isn't an expression.
    """
    try:
        tree = ast.parse(code)
        if not tree.body:
            return None

        last_node = tree.body[-1]
        if not isinstance(last_node, ast.Expr):
            return None

        # Build preceding code from all nodes except the last
        if len(tree.body) > 1:
            preceding = ast.unparse(ast.Module(body=tree.body[:-1], type_ignores=[]))
        else:
            preceding = ''

        expr_code = ast.unparse(last_node.value)
        return preceding, expr_code
    except SyntaxError:
        return None


def execute_code(code: str) -> ExecuteResponse:
    """Execute code in persistent scope.

    Returns response dict with stdout, stderr, result, error fields.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            split = _split_last_expression(code)

            if split is not None:
                preceding, expr_code = split

                if preceding.strip():
                    exec(preceding, _scope_globals)

                result = eval(expr_code, _scope_globals)

                return {
                    'stdout': stdout_capture.getvalue(),
                    'stderr': stderr_capture.getvalue(),
                    'result': repr(result) if result is not None else '',
                    'error': None,
                    'error_type': None,
                    'module_name': None,
                }
            else:
                # Pure statements - just exec
                exec(code, _scope_globals)

                return {
                    'stdout': stdout_capture.getvalue(),
                    'stderr': stderr_capture.getvalue(),
                    'result': '',
                    'error': None,
                    'error_type': None,
                    'module_name': None,
                }

    except ModuleNotFoundError as e:
        return {
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
            'result': '',
            'error': traceback.format_exc(),
            'error_type': 'ModuleNotFoundError',
            'module_name': e.name,
        }
    except Exception:  # exception_safety_linter.py: swallowed-exception — catch-all for arbitrary user code execution; error captured in ExecuteResponse
        return {
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
            'result': '',
            'error': traceback.format_exc(),
            'error_type': None,
            'module_name': None,
        }


def main() -> None:
    """Main driver loop - read requests, execute, send responses."""
    # Signal ready with Python version info
    send_response(
        ReadyResponse(
            status='ready',
            python_version=sys.version,
            python_executable=sys.executable,
        ),
    )

    while True:
        request = read_request()

        if request is None:
            # EOF or protocol error - clean shutdown
            break

        action = request.get('action')

        if action == 'execute':
            code = request.get('code', '')
            response = execute_code(code)
            send_response(response)

        elif action == 'shutdown':
            send_response(ShutdownResponse(status='shutdown'))
            break

        else:
            send_response(
                ErrorResponse(
                    error=f'UnknownAction: {action}',
                    traceback=None,
                ),
            )


if __name__ == '__main__':
    main()
