"""Dashboard launcher - ensures dashboard is running."""

from __future__ import annotations

import logging
import subprocess
import sys
import time

from document_search.dashboard.state import DashboardStateManager

__all__ = [
    'ensure_dashboard',
]

logger = logging.getLogger(__name__)


def ensure_dashboard() -> int:
    """Ensure dashboard is running, return its port.

    Uses file locking to prevent TOCTOU race when multiple MCP servers start simultaneously.
    """
    state_manager = DashboardStateManager()

    with state_manager.hold_lock():
        port = state_manager.get_dashboard_port()
        if port is not None:
            logger.debug(f'Dashboard already running on port {port}')
            return port

        _spawn_dashboard()

    return _wait_for_dashboard(state_manager, timeout=10)


def _spawn_dashboard() -> None:
    """Spawn dashboard as detached process."""
    logger.info('Starting dashboard')

    subprocess.Popen(
        [sys.executable, '-m', 'document_search.dashboard'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _wait_for_dashboard(state_manager: DashboardStateManager, timeout: float) -> int:
    """Wait for dashboard to create state file, return port."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        port = state_manager.get_dashboard_port()
        if port is not None:
            return port
        time.sleep(0.1)
    raise TimeoutError('Dashboard failed to start')
