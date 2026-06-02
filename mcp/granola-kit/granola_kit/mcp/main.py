from __future__ import annotations

__all__ = [
    'main',
    'server',
]

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from granola_kit.clients.granola_api import granola_api_client
from granola_kit.mcp.state import ServerState
from granola_kit.services.meetings import MeetingService
from granola_kit.tools import register_all_tools


@asynccontextmanager
async def lifespan(server_instance: FastMCP) -> AsyncIterator[None]:
    """Build the shared HTTP client + services, register tools, tear everything down on exit."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    async with granola_api_client() as client:
        register_all_tools(ServerState(meetings=MeetingService(client)), server_instance)
        yield


server = FastMCP('granola-kit', lifespan=lifespan)


def main() -> None:
    """Run the granola-kit MCP server."""
    server.run()
