from __future__ import annotations

__all__ = [
    'McpServerInfo',
    'find_live_sock_path',
    'find_one',
    'read_all',
    'register',
    'register_self',
]

import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from cc_lib import os_process
from cc_lib.claude_context import ClaudeContext
from cc_lib.os_process import ProcessHandle
from cc_lib.schemas.base import ClosedModel
from cc_lib.types import CCVersion, JsonDatetime
from cc_lib.utils import get_claude_workspace_config_home_dir
from cc_lib.utils.atomic_write import atomic_write


class McpServerInfo(ClosedModel):
    """Runtime identity of one MCP server process.

    Written to ``~/.claude-workspace/mcp/registry/<session_id>/<name>-<mcp_pid>.json``
    on startup, removed on shutdown.
    """

    __strict_typing_linter__hashable_fields__ = True

    name: str
    """Registered MCP name, e.g. ``'selenium-browser'``."""

    mcp_pid: int
    """The MCP server process's own PID."""

    claude_pid: int
    """The parent Claude Code process PID."""

    session_id: str
    """The Claude session the server belongs to."""

    claude_version: CCVersion
    """Claude Code version, e.g. ``'2.1.138'``."""

    created_at: JsonDatetime
    """The MCP process's create_time, paired with ``mcp_pid`` as a recycle-safe liveness
    anchor (``os_process.ProcessHandle``): a reused PID with a drifted create_time reads as dead."""

    sock_path: str | None = None
    """UDS socket path, set when the server runs a bridge."""

    capabilities: tuple[str, ...] = ()
    """Capability tags, e.g. ``('bridge',)``."""


@asynccontextmanager
async def register(info: McpServerInfo) -> AsyncIterator[McpServerInfo]:
    """Write ``info`` to the registry for the body's duration; remove on exit."""
    path = _entry_path(info.session_id, info.name, info.mcp_pid)
    atomic_write(path, info.model_dump_json().encode())
    try:
        yield info
    finally:
        path.unlink(missing_ok=True)


@asynccontextmanager
async def register_self(
    server: FastMCP,
    *,
    claude_context: ClaudeContext,
    sock_path: str | None = None,
    capabilities: Sequence[str] = (),
) -> AsyncIterator[McpServerInfo]:
    """Register this MCP server under ``server.name`` for the body's duration.

    Reads identity from ``server.name`` (the FastMCP protocol-level name) so
    each MCP declares its name in one place: the ``FastMCP(...)`` constructor.
    """
    pid = os.getpid()
    info = McpServerInfo(
        name=server.name,
        mcp_pid=pid,
        created_at=os_process.create_time(pid),
        claude_pid=claude_context.claude_pid,
        session_id=claude_context.session_id,
        claude_version=claude_context.claude_version,
        sock_path=sock_path,
        capabilities=tuple(capabilities),
    )
    async with register(info):
        yield info


def find_one(session_id: str, name: str) -> McpServerInfo | None:
    """Return the live MCP server entry for ``(session_id, name)``, if any.

    Short-circuits on the first live match; dead-PID entries with the same
    name are skipped.
    """
    registry_dir = _session_dir(session_id)
    if not registry_dir.is_dir():
        return None
    for entry_path in sorted(registry_dir.glob(f'{name}-*.json'), key=_pid_from_entry_path):
        info = McpServerInfo.model_validate_json(entry_path.read_bytes())
        if ProcessHandle(info.mcp_pid, info.created_at).is_alive():
            return info
    return None


def find_live_sock_path(name: str) -> Path | None:
    """Return UDS path for live MCP ``name`` in the current session, or None.

    Read-side counterpart to ``register_self(..., sock_path=...)``: resolves
    the current session via ``ClaudeContext.from_env()`` and reads the
    registry entry. Returns None when no live entry has a ``sock_path``.
    """
    session_id = ClaudeContext.from_env().session_id
    entry = find_one(session_id, name)
    if entry is None or entry.sock_path is None:
        return None
    return Path(entry.sock_path)


def read_all(session_id: str) -> Sequence[McpServerInfo]:
    """Return the MCP servers registered for a session whose PID is alive.

    Entries with a dead PID (crashed server, stale file) are skipped.
    """
    registry_dir = _session_dir(session_id)
    if not registry_dir.is_dir():
        return []
    live: list[McpServerInfo] = []
    for entry in sorted(registry_dir.glob('*.json')):
        info = McpServerInfo.model_validate_json(entry.read_bytes())
        if ProcessHandle(info.mcp_pid, info.created_at).is_alive():
            live.append(info)
    return live


def _pid_from_entry_path(entry_path: Path) -> int:
    return int(entry_path.stem.rsplit('-', 1)[1])


def _session_dir(session_id: str) -> Path:
    return get_claude_workspace_config_home_dir() / 'mcp' / 'registry' / session_id


def _entry_path(session_id: str, mcp_name: str, mcp_pid: int) -> Path:
    return _session_dir(session_id) / f'{mcp_name}-{mcp_pid}.json'
