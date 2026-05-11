"""Detect Claude Code process and resolve current session."""

from __future__ import annotations

import json
import os
import subprocess
import time

import pydantic
from cc_lib.session_tracker import SessionDatabase
from cc_lib.utils import get_claude_workspace_config_home_dir

__all__ = [
    'auto_detect_session_id',
    'find_ancestor_claude_pid',
    'resolve_session_id_from_pid',
]


def find_ancestor_claude_pid() -> int | None:
    """Walk process tree to find an ancestor running the Claude Code binary.

    Verifies each candidate via ``codesign``: Anthropic embeds
    ``Identifier=com.anthropic.claude-code`` in the binary's code signature,
    bound to the code-directory hash. The identifier is preserved across
    adhoc re-signing, so this works for both vanilla and binary-patched
    installs.

    Returns the PID if found, None if not running inside Claude Code.
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

    return None


def resolve_session_id_from_pid(claude_pid: int, *, max_attempts: int) -> str | None:
    """Look up session ID for a Claude PID in sessions.json.

    Returns session ID if found, None if no matching active session.

    Raises:
        RuntimeError: If multiple active sessions match the same PID.
    """
    sessions_file = get_claude_workspace_config_home_dir() / 'sessions.json'
    adapter = pydantic.TypeAdapter(SessionDatabase)

    for attempt in range(max_attempts):
        if not sessions_file.exists():
            if attempt < max_attempts - 1:
                time.sleep(0.1)
            continue

        with sessions_file.open() as f:
            db = adapter.validate_python(json.load(f))

        matching = [s for s in db.sessions if s.state == 'active' and s.metadata.claude_pid == claude_pid]

        if len(matching) == 1:
            return matching[0].session_id
        if len(matching) > 1:
            ids = [s.session_id for s in matching]
            raise RuntimeError(f'Multiple active sessions for PID {claude_pid}: {ids}')

        if attempt < max_attempts - 1:
            time.sleep(0.1)

    return None


def auto_detect_session_id() -> str | None:
    """Find Claude ancestor and resolve session ID.

    Returns session ID if running inside Claude Code, None otherwise.
    """
    pid = find_ancestor_claude_pid()
    if pid is None:
        return None
    return resolve_session_id_from_pid(pid, max_attempts=1)


def _is_claude_binary(pid: int) -> bool:
    """True if ``pid`` is running the Claude Code binary, verified via codesign."""
    result = subprocess.run(
        ['codesign', '-dvv', f'+{pid}'],
        check=False,
        capture_output=True,
        text=True,
    )
    # codesign writes the Identifier= line to stderr.
    return 'Identifier=com.anthropic.claude-code' in result.stderr
