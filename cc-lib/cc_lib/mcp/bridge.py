from __future__ import annotations

__all__ = [
    'UdsBridge',
    'start_uds_bridge',
]

import asyncio
from typing import Any, Protocol

import uvicorn

from cc_lib.mcp.socket_name import get_socket_path


class UdsBridge:
    """A running UDS bridge; call ``stop`` on MCP server shutdown."""

    def __init__(self, server: uvicorn.Server, task: asyncio.Task[None]) -> None:
        self._server = server
        self._task = task

    async def stop(self) -> None:
        """Signal the uvicorn server to exit and await the serving task."""
        self._server.should_exit = True
        await self._task


async def start_uds_bridge(app: ASGIApp, mcp_name: str) -> UdsBridge:
    """Serve ``app`` on the MCP server's per-PID Unix domain socket.

    A stale socket file at the path is removed before binding. Returns a
    handle whose ``stop`` must be awaited on server shutdown.
    """
    socket_path = get_socket_path(mcp_name)
    socket_path.unlink(missing_ok=True)
    # log_config=None defers to the host MCP server's logging configuration
    # instead of letting uvicorn install its own and fragment process logging.
    # ws='none' skips loading the WebSocket protocol — the bridge serves HTTP only.
    config = uvicorn.Config(app, uds=str(socket_path), log_config=None, ws='none')
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    return UdsBridge(server, task)


class ASGIApp(Protocol):
    """An ASGI3 application — the callable shape uvicorn serves."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None: ...
