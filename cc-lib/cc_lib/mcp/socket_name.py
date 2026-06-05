from __future__ import annotations

__all__ = [
    'get_socket_path',
]

import os
from pathlib import Path


def get_socket_path(mcp_name: str, mcp_pid: int | None = None) -> Path:
    """Return the UDS socket path for an MCP server.

    Keyed on the MCP server's PID so sibling MCP servers under one Claude
    parent each bind a distinct socket. ``/tmp`` (not the workspace config dir)
    gives the server and an out-of-process CLI the same predictable absolute
    path. No ``session_id`` in the name: ``mcp_pid`` is machine-global, so
    ``name``-``pid`` is already unique — the registry partitions entries by
    session for *listing*; the socket needs only addressability.

    Args:
        mcp_name: The MCP server's registered name (e.g. ``'selenium-browser'``).
        mcp_pid: Target server PID. Defaults to the current process; pass an
            explicit PID to address a different server.
    """
    return Path(f'/tmp/{mcp_name}-{mcp_pid if mcp_pid is not None else os.getpid()}.sock')
