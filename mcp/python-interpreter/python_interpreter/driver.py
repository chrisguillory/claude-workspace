#!/usr/bin/env python3
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
    {"action": "list_vars"}
    {"action": "reset"}
    {"action": "shutdown"}

Response format:
    {"stdout": "...", "stderr": "...", "result": "...", "error": null}
    {"error": "ErrorType: message", "traceback": "..."}

This script is designed to be completely standalone - it cannot import from
the MCP server since it runs in a different Python environment.
"""

from __future__ import annotations

__all__ = ['main']

import ast
import contextlib
import io
import json
import sys
import traceback
from typing import Any

# Persistent scope for this interpreter - survives across requests
_scope_globals: dict[str, Any] = {}


def read_request() -> dict[str, Any] | None:
    """Read a length-prefixed JSON request from stdin.

    Returns:
        Parsed JSON dict, or None if stdin is closed/EOF.
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

        parsed: dict[str, Any] = json.loads(json_data)
        return parsed

    except (ValueError, json.JSONDecodeError) as e:
        send_response({'error': f'ProtocolError: {e}', 'traceback': None})
        return {}  # Empty dict signals protocol error, continue loop


def send_response(response: dict[str, Any]) -> None:
    """Send a length-prefixed JSON response to stdout."""
    json_data = json.dumps(response)
    sys.stdout.write(f'{len(json_data)}\n{json_data}')
    sys.stdout.flush()


def _split_last_expression(code: str) -> tuple[str, str] | None:
    """Split code into preceding statements and final expression.

    Uses AST node line numbers for precise splitting, handling multi-line
    expressions correctly.

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

        # Use AST line numbers for precise splitting (1-indexed)
        code_lines = code.splitlines(keepends=True)
        preceding = ''.join(code_lines[: last_node.lineno - 1])
        expr_code = ast.unparse(last_node.value)
        return preceding, expr_code
    except SyntaxError:
        return None


def execute_code(code: str) -> dict[str, Any]:
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
                }
            else:
                # Pure statements - just exec
                exec(code, _scope_globals)

                return {
                    'stdout': stdout_capture.getvalue(),
                    'stderr': stderr_capture.getvalue(),
                    'result': '',
                    'error': None,
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
    except Exception:
        return {
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
            'result': '',
            'error': traceback.format_exc(),
        }


def list_vars() -> dict[str, Any]:
    """List user-defined variables in scope."""
    user_vars = sorted(name for name in _scope_globals if not name.startswith('__'))

    if not user_vars:
        return {'result': 'No variables defined', 'error': None}

    return {'result': ', '.join(user_vars), 'error': None}


def main() -> None:
    """Main driver loop - read requests, execute, send responses."""
    # Signal ready with Python version info
    send_response(
        {
            'status': 'ready',
            'python_version': sys.version,
            'python_executable': sys.executable,
        }
    )

    while True:
        request = read_request()

        if request is None:
            # EOF - clean shutdown
            break

        if not request:
            # Empty dict from protocol error - already sent error response
            continue

        action = request.get('action')

        if action == 'execute':
            code = request.get('code', '')
            response = execute_code(code)
            send_response(response)

        elif action == 'list_vars':
            response = list_vars()
            send_response(response)

        elif action == 'reset':
            var_count = len([k for k in _scope_globals if not k.startswith('__')])
            _scope_globals.clear()
            send_response({'result': f'Scope cleared ({var_count} items removed)', 'error': None})

        elif action == 'shutdown':
            send_response({'status': 'shutdown'})
            break

        else:
            send_response(
                {
                    'error': f'UnknownAction: {action}',
                    'traceback': None,
                }
            )


if __name__ == '__main__':
    main()
