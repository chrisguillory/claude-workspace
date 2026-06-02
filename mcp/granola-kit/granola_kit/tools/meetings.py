from __future__ import annotations

__all__ = [
    'register_tools',
]

from collections.abc import Sequence

from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from granola_kit.mcp.state import ServerState
from granola_kit.schemas.results import Meeting


def register_tools(state: ServerState, mcp: FastMCP) -> None:
    """Register the meeting tools on the MCP server."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title='List Meetings',
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def list_meetings(limit: int = 20) -> Sequence[Meeting]:
        """List recent Granola meetings, most-recently-updated first.

        Args:
            limit: Max meetings to return (default 20).
        """
        return await state.meetings.list_meetings(limit=limit)
