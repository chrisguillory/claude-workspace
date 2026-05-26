"""ClaudeContext and primitives for resolving the current Claude session."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pydantic

from cc_lib.exceptions import (
    ClaudeContextError,
    ClaudeProcessNotFoundError,
    InactiveSessionError,
    MissingEnvVarError,
    MultipleActiveSessionsForPidError,
    SessionNotFoundError,
)
from cc_lib.session_tracker import SESSIONS_PATH, Session, SessionDatabase
from cc_lib.types import CCVersion

__all__ = [
    'ClaudeContext',
    'cached_sessions_by_pid',
    'find_claude_pid',
    'in_claude_code',
    'lookup_active_session_by_pid',
    'lookup_session_by_id',
]


class ClaudeContext:
    """Resolved Claude Code session context.

    Two invocation contexts, two named constructors:
        from_env       — Bash-tool subprocess (CLAUDE_CODE_SESSION_ID since 2.1.132)
        from_pid_walk  — long-running process spawned by Claude (e.g., MCP server)

    Wraps the persisted Session record from sessions.json. Convenience accessors
    lift the most-read fields up; consumers needing other Session fields read
    `ctx.session.<field>` directly.
    """

    __slots__ = ('_claude_version', '_session')

    @classmethod
    def from_env(cls) -> ClaudeContext:
        """Bash-tool subprocess context. Reads CLAUDE_CODE_SESSION_ID."""
        session_id = os.environ.get('CLAUDE_CODE_SESSION_ID')
        if not session_id:
            raise MissingEnvVarError('CLAUDE_CODE_SESSION_ID')
        return cls(lookup_session_by_id(session_id))

    @classmethod
    def from_pid_walk(cls) -> ClaudeContext:
        """Long-running-process context. Walks parent tree, codesign-verifies."""
        pid = find_claude_pid()
        return cls(lookup_active_session_by_pid(pid))

    def __init__(self, session: Session) -> None:
        if session.state != 'active':
            raise InactiveSessionError(f"Session {session.session_id} is in state {session.state!r}, expected 'active'")
        if session.metadata.claude_version is None:
            raise ClaudeContextError(f'Session {session.session_id} has no claude_version in sessions.json')
        self._session = session
        self._claude_version = session.metadata.claude_version

    @property
    def session(self) -> Session:
        return self._session

    @property
    def claude_pid(self) -> int:
        return self._session.metadata.claude_pid

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def project_dir(self) -> Path:
        return Path(self._session.project_dir)

    @property
    def claude_version(self) -> CCVersion:
        return self._claude_version

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClaudeContext) and other._session == self._session

    def __hash__(self) -> int:
        return hash(self._session)

    def __repr__(self) -> str:
        return (
            f'ClaudeContext(session_id={self.session_id!r}, '
            f'claude_pid={self.claude_pid}, project_dir={str(self.project_dir)!r})'
        )


def in_claude_code() -> bool:
    """True iff this process was launched inside a Claude Code session.

    Claude Code sets ``CLAUDECODE=1`` on every command it invokes — Bash-tool
    subprocesses, hooks, MCP server starts, anywhere a child process can see
    inherited env. Consumers use this to choose context-appropriate behavior
    (e.g., an error renderer might emit an agent-engagement hint when set, a
    user-facing CTA otherwise).
    """
    return os.environ.get('CLAUDECODE') == '1'


def find_claude_pid() -> int:
    """Walk parent process tree; verify each candidate via macOS codesign.

    Anthropic embeds Identifier=com.anthropic.claude-code in the binary's
    code signature, bound to the code-directory hash. Identifier is preserved
    across the patcher's adhoc re-sign. macOS-only.
    """
    current = os.getppid()
    for _ in range(20):
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid='],
            check=False,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            break
        if _is_claude_binary(current):
            return current
        current = int(result.stdout.strip())
    raise ClaudeProcessNotFoundError('No Claude Code process found in parent tree')


def lookup_session_by_id(session_id: str) -> Session:
    """Look up the active Session record by session_id from sessions.json."""
    db = _load_sessions()
    matching = [s for s in db.sessions if s.session_id == session_id]
    if not matching:
        raise SessionNotFoundError(f'Session {session_id} not in {SESSIONS_PATH}')
    session = matching[0]
    if session.state != 'active':
        raise InactiveSessionError(
            f"Session {session_id} in sessions.json but state={session.state!r}, expected 'active'"
        )
    return session


def lookup_active_session_by_pid(claude_pid: int) -> Session:
    """Look up the active Session record matching this claude PID."""
    db = _load_sessions()
    for_pid = [s for s in db.sessions if s.metadata.claude_pid == claude_pid]
    if not for_pid:
        raise SessionNotFoundError(f'No session for PID {claude_pid} in {SESSIONS_PATH}')
    active = [s for s in for_pid if s.state == 'active']
    if not active:
        states = [(s.session_id, s.state) for s in for_pid]
        raise InactiveSessionError(f'Sessions for PID {claude_pid} exist but none are active: {states}')
    if len(active) > 1:
        ids = [s.session_id for s in active]
        raise MultipleActiveSessionsForPidError(f'Multiple active sessions for PID {claude_pid}: {ids}')
    return active[0]


def cached_sessions_by_pid() -> Mapping[int, Session]:
    """Return dict[pid → Session] of active sessions, mtime-cached.

    For high-frequency reverse lookups (mitmproxy attaching session metadata
    to each captured request).
    """
    return _SessionsByPidCache.get()


class _SessionsByPidCache:
    """Module-singleton mtime-invalidated cache of active sessions keyed by claude_pid."""

    _mtime: float | None = None
    _by_pid: Mapping[int, Session] = {}

    @classmethod
    def get(cls) -> Mapping[int, Session]:
        if not SESSIONS_PATH.exists():
            return {}
        mtime = SESSIONS_PATH.stat().st_mtime
        if mtime != cls._mtime:
            db = _load_sessions()
            cls._by_pid = {s.metadata.claude_pid: s for s in db.sessions if s.state == 'active'}
            cls._mtime = mtime
        return cls._by_pid


def _is_claude_binary(pid: int) -> bool:
    """True if pid runs the Claude Code binary (codesign Identifier verified)."""
    result = subprocess.run(
        ['codesign', '-dvv', f'+{pid}'],
        check=False,
        capture_output=True,
        text=True,
    )
    # codesign writes the Identifier= line to stderr.
    return 'Identifier=com.anthropic.claude-code' in result.stderr.splitlines()


def _load_sessions() -> SessionDatabase:
    """Load sessions.json (or return empty database if file missing)."""
    if not SESSIONS_PATH.exists():
        return SessionDatabase(sessions=[])
    adapter = pydantic.TypeAdapter(SessionDatabase)
    with SESSIONS_PATH.open() as f:
        return adapter.validate_python(json.load(f))
