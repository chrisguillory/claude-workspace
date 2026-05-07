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
from dataclasses import dataclass, field
from pathlib import Path

from grok_kit_sdk import models
from mcp.server.fastmcp import FastMCP

from grok_kit.auth import ImportResult, import_state
from grok_kit.service import Conversation, GrokService

logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    """Mutable server state initialized at startup.

    The ``service`` is built lazily on first conversation-tool call so the
    server can start even before cookies have been imported (the
    ``grok_auth_import`` tool is the bootstrap path for that case).
    """

    service: GrokService | None = field(default=None)

    def get_service(self) -> GrokService:
        if self.service is None:
            self.service = GrokService.from_cookies()
        return self.service

    def reset_service(self) -> None:
        """Drop the cached service so the next call rebuilds with fresh cookies."""
        self.service = None


@contextlib.asynccontextmanager
async def lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
    """Initialize logging and register tools."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    state = ServerState()
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
        convs = state.get_service().list_conversations(limit=None if title_contains else limit)
        if title_contains is not None:
            needle = title_contains if case_sensitive else title_contains.lower()
            convs = [c for c in convs if needle in (c.title if case_sensitive else c.title.lower())][:limit]
        return convs

    @server.tool()
    async def grok_get_conversation(conversation_id: str) -> Conversation:
        """Fetch a conversation's metadata, message tree, and full bodies.

        Returns messages ordered chronologically.
        """
        return state.get_service().get_full_conversation(conversation_id)

    @server.tool()
    async def grok_auth_import(state_path: str) -> ImportResult:
        """Import a profile-state JSON into grok-kit's cookie store.

        Reads the file at ``state_path``, validates it as ProfileState, and
        atomically writes to the canonical cookie path. Returns counts and
        load-bearing-cookie status. Subsequent conversation tool calls pick up
        the new cookies automatically.
        """
        result = import_state(Path(state_path))
        state.reset_service()
        return result


def main() -> None:
    """Run the MCP server."""
    server.run()
