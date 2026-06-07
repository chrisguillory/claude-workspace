"""preflight-check MCP server — expose readiness checks to Claude."""

from __future__ import annotations

__all__ = [
    'main',
    'server',
]

import logging
import sys

from mcp.server.fastmcp import FastMCP

from preflight_check.checks import ALL_CHECKS
from preflight_check.report import Report
from preflight_check.runner import run_checks

logger = logging.getLogger(__name__)

server = FastMCP('preflight-check')


@server.tool()
def check() -> Report:
    """Run machine-readiness checks on the current host; return the structured report.

    Prefer the CLI for interactive use:
        preflight-check check
    """
    return run_checks(ALL_CHECKS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    server.run()


if __name__ == '__main__':
    main()
