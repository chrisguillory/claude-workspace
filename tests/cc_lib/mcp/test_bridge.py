"""Tests for cc_lib.mcp.bridge -- start_uds_bridge + UdsBridge lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from cc_lib.mcp.bridge import start_uds_bridge
from cc_lib.mcp.socket_name import get_socket_path


async def test_bridge_binds_and_stops_cleanly() -> None:
    """End-to-end: start, uvicorn binds the per-MCP-PID UDS socket, stop awaits cleanly.

    Async + real-socket coordination is hard to stage empirically without
    reproducing exactly this sequence.
    """
    mcp_name = 'cc-lib-test-bridge'
    socket_path = get_socket_path(mcp_name)
    socket_path.unlink(missing_ok=True)
    try:
        bridge = await start_uds_bridge(_trivial_asgi_app, mcp_name)
        for _ in range(50):
            if socket_path.is_socket():
                break
            await asyncio.sleep(0.02)
        assert socket_path.is_socket(), f'bridge did not bind {socket_path}'
        await bridge.stop()
    finally:
        socket_path.unlink(missing_ok=True)


async def _trivial_asgi_app(
    scope: Mapping[str, Any],
    receive: Callable[[], Awaitable[Mapping[str, Any]]],  # noqa: ARG001 — ASGI contract
    send: Callable[[Mapping[str, Any]], Awaitable[None]],
) -> None:
    """Smallest ASGI3 app — answers any HTTP request with a 200/ok body."""
    if scope['type'] == 'http':
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
        await send({'type': 'http.response.body', 'body': b'ok'})
