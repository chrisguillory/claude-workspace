"""Dashboard web server."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from document_search.dashboard.state import DashboardStateManager
from document_search.schemas.dashboard import DashboardState, McpServer

__all__ = [
    'DashboardServer',
]

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
MONITOR_INTERVAL_SECONDS = 5

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Document Search Dashboard</title>
    <style>
        body { font-family: system-ui, sans-serif; margin: 2rem; }
        .server { padding: 0.5rem; margin: 0.5rem 0; background: #f0f0f0; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>Document Search Dashboard</h1>
    <div id="servers"></div>
    <script>
        async function update() {
            const resp = await fetch('/api/mcp-servers');
            const servers = await resp.json();
            const el = document.getElementById('servers');
            if (servers.length === 0) {
                el.innerHTML = '<p>No MCP servers connected</p>';
            } else {
                el.innerHTML = servers.map(s =>
                    `<div class="server">PID: ${s.pid} | Started: ${s.started_at}</div>`
                ).join('');
            }
        }
        update();
        setInterval(update, 2000);
    </script>
</body>
</html>"""


class DashboardServer:
    """Dashboard server with MCP server monitoring.

    MCP servers register themselves via DashboardStateManager.register_mcp_server().
    Dashboard monitors registered PIDs and exits when none are alive.
    """

    def __init__(self, state_manager: DashboardStateManager) -> None:
        self._state_manager = state_manager
        self._should_exit = False

    async def run(self) -> None:
        """Run server until all MCP servers exit."""
        previous_state = self._state_manager.load()
        existing_servers = previous_state.mcp_servers if previous_state else []
        preferred_port = previous_state.port if previous_state else None

        port = _find_port(preferred_port)

        state = DashboardState(
            port=port,
            server_pid=os.getpid(),
            mcp_servers=existing_servers,
        )
        self._state_manager.save(state)
        logger.info(f'Dashboard starting on http://127.0.0.1:{port}')

        monitor_task = asyncio.create_task(self._monitor_mcp_servers())

        config = uvicorn.Config(
            app=self._create_app(),
            host='127.0.0.1',
            port=port,
            log_level='warning',
        )
        server = uvicorn.Server(config)

        try:
            await server.serve()
        finally:
            monitor_task.cancel()
            self._state_manager.delete()

    async def _monitor_mcp_servers(self) -> None:
        """Shutdown when no MCP servers alive."""
        # Grace period for initial registration
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS * 2)

        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            live = self._state_manager.get_live_mcp_servers()

            if not live:
                logger.info('No live MCP servers, shutting down')
                # Graceful exit - raises SystemExit which uvicorn handles
                raise SystemExit(0)

    def _create_app(self) -> FastAPI:
        """Create FastAPI app."""
        app = FastAPI(title='Document Search Dashboard')
        state_manager = self._state_manager

        @app.get('/api/state')
        def get_state() -> DashboardState:
            state = state_manager.load()
            if state is None:
                raise HTTPException(status_code=503, detail='Dashboard state not available')
            return state

        @app.get('/api/mcp-servers')
        def get_mcp_servers() -> Sequence[McpServer]:
            return state_manager.get_live_mcp_servers()

        @app.get('/', response_class=HTMLResponse)
        def index() -> str:
            return INDEX_HTML

        return app


def _find_port(preferred: int | None) -> int:
    """Find available port. Tries preferred, then default, then OS-assigned."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    if preferred != DEFAULT_PORT:
        candidates.append(DEFAULT_PORT)

    for port in candidates:
        if _port_available(port):
            return port

    return _get_free_port()


def _port_available(port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _get_free_port() -> int:
    """Get OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port: int = s.getsockname()[1]
        return port
