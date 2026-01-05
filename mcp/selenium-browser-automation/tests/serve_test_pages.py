#!/usr/bin/env python3
"""
Start HTTP servers on multiple ports for multi-origin localStorage testing.

Usage:
    python tests/serve_test_pages.py

Starts servers on ports 8001, 8002, 8003, each serving tests/fixtures/.
Each port is a different origin for localStorage isolation testing.

Press Ctrl+C to stop all servers.
"""

from __future__ import annotations

import http.server
import os
import threading
from pathlib import Path

PORTS = [8001, 8002, 8003]
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def start_server(port: int) -> http.server.HTTPServer:
    """Start HTTP server on given port, serving FIXTURES_DIR."""
    os.chdir(FIXTURES_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(('localhost', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    print(f'Serving {FIXTURES_DIR}')
    print()

    servers = []
    for port in PORTS:
        server = start_server(port)
        servers.append(server)
        print(f'  http://localhost:{port}/storage-test-page.html')

    print()
    print('Press Ctrl+C to stop all servers...')
    print()

    try:
        # Keep main thread alive
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\nStopping servers...')
        for server in servers:
            server.shutdown()
        print('Done.')


if __name__ == '__main__':
    main()
