#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["httpx"]
# ///
"""
HTTP client for python-interpreter MCP server.

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

__all__ = ['main']

import argparse
import os
import pathlib
import sys

import httpx


def get_socket_path() -> pathlib.Path:
    """Get Unix socket path by finding Claude PID in process tree."""
    import subprocess

    current = os.getppid()

    for _ in range(20):  # Depth limit
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid=,comm='],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            break

        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ''

        # Check if this is Claude
        if 'claude' in comm.lower():
            return pathlib.Path(f'/tmp/python-interpreter-{current}.sock')

        if ppid == 0:
            break

        current = ppid

    raise RuntimeError('Could not find Claude process in process tree')


def main() -> None:
    parser = argparse.ArgumentParser(description='Execute Python code via MCP interpreter')
    parser.add_argument('--interpreter', '-i', default='builtin', help='Interpreter name (default: builtin)')
    args = parser.parse_args()

    # Read code from stdin
    code = sys.stdin.read()
    if not code.strip():
        print('Error: No code provided on stdin', file=sys.stderr)
        sys.exit(1)

    # Get socket path
    socket_path = get_socket_path()

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
            except Exception:
                # Fallback if response isn't JSON
                print(f'Error: {response.text}', file=sys.stderr)

            response.raise_for_status()

        # Print result
        result = response.json()['result']
        print(result)


if __name__ == '__main__':
    main()
