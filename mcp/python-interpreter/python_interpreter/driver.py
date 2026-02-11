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


def detect_last_expression(code: str) -> tuple[bool, str | None]:
    """Detect if the last statement is an expression.

    Returns:
        (is_expression, unparsed_last_line) or (False, None)
    """
    try:
        tree = ast.parse(code)
        if not tree.body:
            return False, None

        last_node = tree.body[-1]
        if isinstance(last_node, ast.Expr):
            return True, ast.unparse(last_node.value)
        return False, None
    except SyntaxError:
        return False, None


def execute_code(code: str) -> dict[str, Any]:
    """Execute code in persistent scope.

    Returns response dict with stdout, stderr, result, error fields.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            is_expr, last_line = detect_last_expression(code)

            if is_expr and last_line:
                # Execute all but last line, then eval last line for its value
                lines = code.splitlines()
                code_without_last = '\n'.join(lines[:-1])

                if code_without_last.strip():
                    exec(code_without_last, _scope_globals)

                result = eval(last_line, _scope_globals)

                stdout_val = stdout_capture.getvalue()
                stderr_val = stderr_capture.getvalue()

                # If there was print output, return that; otherwise return repr of result
                if stdout_val or stderr_val:
                    return {
                        'stdout': stdout_val,
                        'stderr': stderr_val,
                        'result': '',
                        'error': None,
                    }
                else:
                    return {
                        'stdout': '',
                        'stderr': '',
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
