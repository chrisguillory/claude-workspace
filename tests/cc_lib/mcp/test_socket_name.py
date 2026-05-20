"""Tests for cc_lib.mcp.socket_name -- per-MCP-PID UDS socket paths."""

from __future__ import annotations

from pathlib import Path

from cc_lib.mcp.socket_name import get_socket_path


def test_zero_pid_treated_as_explicit() -> None:
    """``mcp_pid=0`` must use 0, not fall through to os.getpid().

    Pins the ``is not None`` check (regression bait if rewritten to a truthy
    check like ``mcp_pid or os.getpid()``). Also pins the path format —
    empirical adoption catches happy-path format breaks, this catches the
    None-vs-falsy edge.
    """
    assert get_socket_path('x', mcp_pid=0) == Path('/tmp/x-0.sock')
