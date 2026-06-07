#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "httpx",
# ]
# ///
"""HTTP client for python-interpreter MCP server.

Connects to the MCP server via Unix socket.

Usage Examples:

1. Simple expression:
   $ uv run --script python-interpreter-client.py <<'PY'
   import pandas as pd
   print(pd.__version())
   PY

   OUTPUT:
   2.1.4

2. Large output (gets truncated):
   $ uv run --script python-interpreter-client.py <<'PY'
   print('x' * 30000)
   PY

   OUTPUT:
   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx... [25,000 chars shown]

   ==================================================
   # OUTPUT TRUNCATED
   # Original size: 30,000 chars
   # Full output: /tmp/tmpXYZ123/output_20250108_143022.txt
   ==================================================
"""

from __future__ import annotations

import argparse
import sys

import httpx
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.mcp import find_live_sock_path

from python_interpreter import PROJECT

boundary = ErrorBoundary(exit_code=1)


@boundary
def main() -> None:
    parser = argparse.ArgumentParser(description='Execute Python code via MCP interpreter')
    parser.add_argument('--interpreter', '-i', default='builtin', help='Interpreter name (default: builtin)')
    args = parser.parse_args()

    # Read code from stdin
    code = sys.stdin.read()
    if not code.strip():
        print('Error: No code provided on stdin', file=sys.stderr)
        sys.exit(1)

    socket_path = find_live_sock_path(PROJECT.name)
    if socket_path is None:
        print(f'Error: no live {PROJECT.name} MCP for current session', file=sys.stderr)
        print(f'Is the {PROJECT.name} MCP server running?', file=sys.stderr)
        sys.exit(1)

    # Build request payload
    payload = {'code': code, 'interpreter': args.interpreter}

    # Connect via Unix socket
    transport = httpx.HTTPTransport(uds=socket_path.as_posix())
    with httpx.Client(transport=transport, timeout=60.0) as client:
        response = client.post('http://localhost/execute', json=payload)

        # Handle errors by printing response details before raising
        if response.is_error:
            try:
                error_data = response.json()
                if 'detail' in error_data:
                    print(error_data['detail'], file=sys.stderr)
                if 'traceback' in error_data:
                    print(error_data['traceback'], file=sys.stderr)
            except (ValueError, KeyError):
                # Fallback if response isn't JSON or missing expected keys
                print(f'Error: {response.text}', file=sys.stderr)

            response.raise_for_status()

        # Print result
        result = response.json()['result']
        print(result)


if __name__ == '__main__':
    main()
