"""
Selenium Browser Automation MCP Server

CDP stealth injection to bypass Cloudflare bot detection.

Install:
    uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser-automation mcp-selenium-browser-automation-server

Architecture: Runs locally (not Docker) for visible browser monitoring.
Uses Selenium with CDP stealth injection where Playwright fails.
"""

from __future__ import annotations

# Standard Library
import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import types
import typing

# Third-Party Libraries
from mcp.server.fastmcp import FastMCP

# Local
# Local imports
from .service import BrowserService
from .state import BrowserState
from .tools import register_all_tools

__all__ = [
    'lifespan',
    'main',
    'register_tools',
]


def register_tools(service: BrowserService) -> None:
    """Register all MCP tools — delegates to tools/ package."""
    register_all_tools(service, mcp)


@contextlib.asynccontextmanager
async def lifespan(server_instance: FastMCP) -> typing.AsyncIterator[None]:
    """Manage browser lifecycle - initialization before requests, cleanup after shutdown."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )

    state = await BrowserState.create()
    service = BrowserService(state)
    register_tools(service)

    # Register signal handlers to ensure cleanup on SIGTERM/SIGINT
    # This is critical for `claude mcp reconnect` which sends SIGTERM
    def signal_handler(signum: int, frame: types.FrameType | None) -> None:
        _sync_cleanup(state)
        sys.stderr.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info('Browser service initialized (screenshots: %s, captures: %s)', state.screenshot_dir, state.capture_dir)

    yield

    # SHUTDOWN: Cleanup after all requests complete (graceful shutdown path)
    if state.driver:
        await asyncio.to_thread(state.driver.quit)
    if state.mitmproxy_process:
        state.mitmproxy_process.terminate()
        try:
            state.mitmproxy_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.mitmproxy_process.kill()
    state.temp_dir.cleanup()
    state.capture_temp_dir.cleanup()
    logger.info('Server cleanup complete')


mcp = FastMCP('selenium-browser-automation', lifespan=lifespan)


def main() -> None:
    """Entry point for uvx installation."""
    logger.info('Starting Selenium Browser Automation MCP server (CDP stealth injection for bot detection bypass)')
    mcp.run()


logger = logging.getLogger(__name__)


def _sync_cleanup(state: BrowserState, timeout: int = 5) -> None:
    """Synchronous cleanup for signal handlers (runs in main thread).

    Uses a thread with timeout to prevent hanging on unresponsive ChromeDriver.
    """
    print('\n⚠ Signal received, cleaning up browser...', file=sys.stderr)
    if state.driver:
        try:
            # driver.quit() sends an HTTP request to ChromeDriver which can hang
            # if Chrome/ChromeDriver is unresponsive. Run in a thread with timeout.
            quit_thread = threading.Thread(target=state.driver.quit, daemon=True)
            quit_thread.start()
            quit_thread.join(timeout=timeout)
            if quit_thread.is_alive():
                print(f'✗ Browser close timed out after {timeout}s, abandoning', file=sys.stderr)
            else:
                print('✓ Browser closed', file=sys.stderr)
        except Exception as e:  # exception_safety_linter.py: swallowed-exception — signal cleanup must not raise
            print(f'✗ Browser close error: {e}', file=sys.stderr)
    if state.mitmproxy_process:
        state.mitmproxy_process.terminate()
        try:
            state.mitmproxy_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            state.mitmproxy_process.kill()
    try:
        state.temp_dir.cleanup()
        state.capture_temp_dir.cleanup()
    except Exception:  # exception_safety_linter.py: swallowed-exception — signal cleanup must not raise
        pass
    print('✓ Signal cleanup complete, exiting', file=sys.stderr)


if __name__ == '__main__':
    main()
