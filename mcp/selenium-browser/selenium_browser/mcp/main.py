"""
Selenium Browser Automation MCP Server

CDP stealth injection to bypass Cloudflare bot detection.

Install:
    uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser selenium-browser-mcp

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
from pathlib import Path

# Third-Party Libraries
import uvicorn
from mcp.server.fastmcp import FastMCP

# Local
from ..bridge import create_bridge_app
from ..service import BrowserService
from ..state import BrowserState
from ..tools import register_all_tools

__all__ = [
    'lifespan',
    'main',
]


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
    _register_tools(service)

    # Register signal handlers to ensure cleanup on SIGTERM/SIGINT
    # This is critical for `claude mcp reconnect` which sends SIGTERM
    def signal_handler(signum: int, frame: types.FrameType | None) -> None:
        _sync_cleanup(state)
        sys.stderr.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start HTTP bridge on Unix socket for CLI client access
    socket_path = _find_socket_path()
    uvicorn_server = None
    uvicorn_task = None
    if socket_path:
        # Clean stale socket from previous run
        if socket_path.exists():
            socket_path.unlink()

        state.cli_lock = asyncio.Lock()
        bridge_app = create_bridge_app(service, state.cli_lock)
        config = uvicorn.Config(bridge_app, uds=socket_path.as_posix(), log_level='warning')
        uvicorn_server = uvicorn.Server(config)
        uvicorn_task = asyncio.create_task(uvicorn_server.serve())
        logger.info('HTTP bridge: %s', socket_path)

    logger.info('Browser service initialized (screenshots: %s, captures: %s)', state.screenshot_dir, state.capture_dir)

    yield

    # SHUTDOWN: Stop HTTP bridge
    if uvicorn_server:
        uvicorn_server.should_exit = True
    if uvicorn_task:
        uvicorn_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await uvicorn_task
    if socket_path and socket_path.exists():
        socket_path.unlink()

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


mcp = FastMCP('selenium-browser', lifespan=lifespan)


def main() -> None:
    """Entry point for uvx installation."""
    logger.info('Starting Selenium Browser Automation MCP server (CDP stealth injection for bot detection bypass)')
    mcp.run()


logger = logging.getLogger(__name__)


def _register_tools(service: BrowserService) -> None:
    """Register all MCP tools — delegates to tools/ package."""
    register_all_tools(service, mcp)


def _find_socket_path() -> Path | None:
    """Find Claude Code PID via process tree walk, return socket path.

    Returns None if not running under Claude Code (e.g., standalone testing).
    """
    current = os.getppid()
    for _ in range(20):
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid=,comm='],
            check=False,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            break
        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ''
        if 'claude' in comm.lower():
            return Path(f'/tmp/selenium-browser-{current}.sock')
        if ppid == 0:
            break
        current = ppid
    logger.info('Claude Code not found in process tree — HTTP bridge disabled')
    return None


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
