"""Dashboard coordination schemas.

Shared data models for MCP server registration and dashboard state.
Used by both the dashboard server and MCP server for coordination.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_lib.types import JsonDatetime

from document_search.schemas.base import StrictModel

__all__ = [
    'DashboardState',
    'McpServer',
]


class McpServer(StrictModel):
    """A registered MCP server instance.

    Identified by PID with started_at for robustness against PID reuse.
    """

    pid: int
    started_at: JsonDatetime


class DashboardState(StrictModel):
    """Persisted dashboard coordination state.

    Tracks the dashboard server and registered MCP servers.
    File location: paths.DASHBOARD_STATE_PATH
    """

    port: int
    server_pid: int
    mcp_servers: Sequence[McpServer]
