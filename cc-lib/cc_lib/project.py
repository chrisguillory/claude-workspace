"""Static identity of a workspace project (MCP server, CLI, daemon).

Companion to ``cc_lib.mcp.McpServerInfo``: ``Project`` is what the package
declares about itself before any process starts; ``McpServerInfo`` is what
the running process registers (pid, session, socket, capabilities).
"""

from __future__ import annotations

__all__ = [
    'Project',
]

from cc_lib.schemas.base import ClosedModel


class Project(ClosedModel):
    """A workspace project's static identity."""

    name: str
    """Kebab-case canonical identifier; mirrors ``[project] name`` in pyproject.toml."""
