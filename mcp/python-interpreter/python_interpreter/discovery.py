"""Claude Code environment discovery.

Discovers the Claude Code process and session context by walking the process tree
and reading from claude-workspace session tracking.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib
import subprocess
import time

from cc_lib.session_tracker import SESSIONS_PATH, SessionDatabase
from cc_lib.utils import get_claude_config_home_dir

__all__ = [
    'ClaudeContext',
    'discover_session_id',
    'find_claude_context',
]


@dataclasses.dataclass(frozen=True, slots=True)
class ClaudeContext:
    """Context information about the Claude Code session."""

    claude_pid: int
    project_dir: pathlib.Path
    socket_path: pathlib.Path


def discover_session_id(claude_pid: int) -> str:
    """Discover Claude Code session ID from claude-workspace sessions.json.

    Looks up the active session matching the given Claude PID in the sessions
    database maintained by claude-workspace hooks.

    Related: https://github.com/anthropics/claude-code/issues/1335
             https://github.com/anthropics/claude-code/issues/1407
             https://github.com/anthropics/claude-code/issues/5262

    Args:
        claude_pid: PID of the Claude process (from find_claude_context)

    Returns:
        Session ID (UUID string)

    Raises:
        RuntimeError: If sessions.json doesn't exist, no matching session found,
                      or multiple active sessions match the PID
    """
    max_retries = 20
    retry_delay = 0.05  # 50ms between retries

    for _ in range(max_retries):
        if not SESSIONS_PATH.exists():
            time.sleep(retry_delay)
            continue

        with SESSIONS_PATH.open() as f:
            data = json.load(f)

        db = SessionDatabase.model_validate(data)

        # Find active sessions matching our Claude PID
        matching = [s for s in db.sessions if s.state == 'active' and s.metadata.claude_pid == claude_pid]

        if len(matching) == 1:
            return matching[0].session_id

        if len(matching) > 1:
            session_ids = [s.session_id for s in matching]
            raise RuntimeError(f'Multiple active sessions found for Claude PID {claude_pid}: {session_ids}')

        # No match yet - hook may still be writing
        time.sleep(retry_delay)

    raise RuntimeError(
        f'Could not find active session for Claude PID {claude_pid} in {SESSIONS_PATH} '
        f'after {max_retries} attempts. Ensure claude-workspace SessionStart hook is configured.',
    )


@functools.cache
def find_claude_context() -> ClaudeContext:
    """Find Claude process and extract its context (PID, project directory).

    Walks the process tree upward from the current process to find the Claude Code
    process. Verifies by checking for open .claude/ files via lsof.

    Returns:
        ClaudeContext with claude_pid, project_dir, and socket_path

    Raises:
        RuntimeError: If not running under Claude Code or cannot determine CWD
    """
    current = os.getppid()

    for _ in range(20):  # Depth limit
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
            # Get Claude's CWD using lsof -F n (machine-parseable, handles paths with spaces)
            result = subprocess.run(
                ['lsof', '-p', str(current), '-a', '-d', 'cwd', '-F', 'n'],
                check=False,
                capture_output=True,
                text=True,
            )

            cwd = None
            for line in result.stdout.splitlines():
                if line.startswith('n'):
                    cwd = pathlib.Path(line[1:])  # Strip 'n' prefix
                    break

            if not cwd:
                raise RuntimeError(f'Found Claude process (PID {current}) but could not determine CWD')

            # Verify by checking if Claude has config dir files open
            result = subprocess.run(
                ['lsof', '-p', str(current), '-F', 'n'],
                check=False,
                capture_output=True,
                text=True,
            )

            claude_dir = get_claude_config_home_dir()
            claude_dir_name = claude_dir.name
            claude_files = [
                pathlib.Path(line[1:])
                for line in result.stdout.splitlines()
                if line.startswith('n') and claude_dir_name in line
            ]

            if not claude_files:
                raise RuntimeError(
                    f'Found Claude process (PID {current}) with CWD {cwd}, '
                    f'but no {claude_dir_name}/ files are open - may not be a Claude project',
                )

            # Verify at least one file is in the Claude config directory
            matching_files = [f for f in claude_files if f.is_relative_to(claude_dir)]

            if not matching_files:
                raise RuntimeError(
                    f'Found Claude process (PID {current}) with CWD {cwd}, '
                    f'but {claude_dir_name}/ files open are not in {claude_dir}/ directory:\n'
                    f'  Open files: {claude_files}\n'
                    f'  Expected to find files in: {claude_dir}',
                )

            socket_path = pathlib.Path(f'/tmp/python-interpreter-{current}.sock')
            return ClaudeContext(claude_pid=current, project_dir=cwd, socket_path=socket_path)

        if ppid == 0:
            break

        current = ppid

    raise RuntimeError('Not running under Claude Code - could not find Claude process in process tree')
