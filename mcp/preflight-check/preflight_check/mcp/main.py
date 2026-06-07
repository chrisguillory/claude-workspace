"""preflight-check MCP server — expose readiness checks to Claude."""

from __future__ import annotations

__all__ = [
    'main',
    'server',
]

import contextlib
import logging
from collections.abc import AsyncIterator

from cc_lib.claude_context import ClaudeContext
from cc_lib.logging_setup import configure_logging
from cc_lib.mcp import register_self
from mcp.server.fastmcp import FastMCP

from preflight_check import PROJECT
from preflight_check.checks import ALL_CHECKS
from preflight_check.report import Report
from preflight_check.runner import run_checks

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(mcp_server: FastMCP) -> AsyncIterator[None]:
    configure_logging()
    claude_context = ClaudeContext.from_pid_walk()
    async with register_self(mcp_server, claude_context=claude_context, sock_path=None, capabilities=()):
        logger.info('preflight-check MCP ready (session %s)', claude_context.session_id)
        yield


server = FastMCP(PROJECT.name, lifespan=lifespan)


@server.tool()
def check() -> Report:
    """Run machine-readiness checks on the current host; return the structured report.

    Prefer the CLI for interactive use:
        preflight-check check
    """
    return run_checks(ALL_CHECKS)


def main() -> None:
    server.run()


if __name__ == '__main__':
    main()
