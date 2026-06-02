from __future__ import annotations

__all__ = [
    'register_all_tools',
]

from fastmcp import FastMCP

from granola_kit.mcp.state import ServerState
from granola_kit.tools import meetings


def register_all_tools(state: ServerState, mcp: FastMCP) -> None:
    """Register all granola-kit MCP tools from their per-domain modules."""
    meetings.register_tools(state, mcp)
