"""grok-kit MCP server — exposes GrokService over MCP."""

from __future__ import annotations

__all__ = [
    'ServerState',
    'lifespan',
    'main',
    'register_tools',
    'server',
]

import contextlib
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from grok_kit_sdk import models
from mcp.server.fastmcp import FastMCP

from grok_kit.service import Conversation, GrokService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerState:
    """Immutable server state initialized at startup."""

    service: GrokService


@contextlib.asynccontextmanager
async def lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
    """Construct GrokService from cookies on disk and register tools."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    state = ServerState(service=GrokService.from_cookies())
    register_tools(state)
    logger.info('grok-kit MCP server ready')
    yield


server = FastMCP('grok-kit', lifespan=lifespan)


def register_tools(state: ServerState) -> None:
    """Register MCP tools with closure over server state."""

    @server.tool()
    async def grok_list_conversations(
        title_contains: str | None = None,
        case_sensitive: bool = False,
        limit: int = 60,
    ) -> Sequence[models.ConversationSummary]:
        """List grok.com conversations.

        Server returns most-recently-modified first. ``title_contains`` filters
        client-side; the underlying API has no text-search parameter.
        """
        convs = state.service.list_conversations(limit=None if title_contains else limit)
        if title_contains is not None:
            needle = title_contains if case_sensitive else title_contains.lower()
            convs = [c for c in convs if needle in (c.title if case_sensitive else c.title.lower())][:limit]
        return convs

    @server.tool()
    async def grok_get_conversation(
        conversation_id: str,
    ) -> Conversation:
        """Fetch a conversation's metadata, message tree, and full bodies.

        Returns messages ordered chronologically.
        """
        return state.service.get_full_conversation(conversation_id)


def main() -> None:
    """Run the MCP server."""
    server.run()
