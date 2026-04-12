from __future__ import annotations

__all__ = [
    'register_all_tools',
]

from mcp.server.fastmcp import FastMCP

from ..service import BrowserService
from . import browser_management, extraction, interaction, navigation, performance, profile_state, proxy, waiting


def register_all_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register all MCP tools from domain-grouped modules."""
    navigation.register_tools(service, mcp)
    extraction.register_tools(service, mcp)
    interaction.register_tools(service, mcp)
    waiting.register_tools(service, mcp)
    performance.register_tools(service, mcp)
    profile_state.register_tools(service, mcp)
    proxy.register_tools(service, mcp)
    browser_management.register_tools(service, mcp)
