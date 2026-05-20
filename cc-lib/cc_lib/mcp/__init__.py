"""MCP server identity registry and per-MCP-PID UDS sockets."""

from __future__ import annotations

from cc_lib.mcp.server_registry import (
    McpServerInfo,
    find_live_sock_path,
    find_one,
    read_all,
    register,
    register_self,
)
from cc_lib.mcp.socket_name import get_socket_path

__all__ = [
    'McpServerInfo',
    'find_live_sock_path',
    'find_one',
    'get_socket_path',
    'read_all',
    'register',
    'register_self',
]
