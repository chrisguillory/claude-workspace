from __future__ import annotations

__all__ = [
    'register_all_tools',
]

from mcp.server.fastmcp import FastMCP

from ..service import BrowserService
from .browser_mgmt import register_browser_mgmt_tools
from .extraction import register_extraction_tools
from .interaction import register_interaction_tools
from .navigation import register_navigation_tools
from .performance import register_performance_tools
from .profile_state import register_profile_state_tools
from .proxy import register_proxy_tools
from .waiting import register_waiting_tools


def register_all_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register all MCP tools from domain-grouped modules."""
    register_navigation_tools(service, mcp)
    register_extraction_tools(service, mcp)
    register_interaction_tools(service, mcp)
    register_waiting_tools(service, mcp)
    register_performance_tools(service, mcp)
    register_profile_state_tools(service, mcp)
    register_proxy_tools(service, mcp)
    register_browser_mgmt_tools(service, mcp)
