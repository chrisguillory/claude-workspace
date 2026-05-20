"""Regression tests for DashboardStateManager liveness checks."""

from __future__ import annotations

import os
import socket
from pathlib import Path

from document_search.dashboard.state import DashboardStateManager
from document_search.schemas.dashboard import DashboardState


def test_get_dashboard_port_returns_none_when_pid_alive_but_port_unbound(
    tmp_path: Path,
) -> None:
    """Catch the PID-reuse failure mode.

    Stale state with a recycled-but-live PID and an unbound port must not
    masquerade as a running dashboard. PID-only liveness fails this case;
    the port probe is what catches it.
    """
    manager = DashboardStateManager(
        state_path=tmp_path / 'dashboard.json',
        lock_path=tmp_path / 'dashboard.lock',
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        free_port = probe.getsockname()[1]

    manager.save(
        DashboardState(port=free_port, server_pid=os.getpid(), registered_processes=[]),
    )

    assert manager.get_dashboard_port() is None
