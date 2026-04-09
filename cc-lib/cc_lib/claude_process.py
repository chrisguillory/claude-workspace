from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence

from cc_lib.session_tracker import find_claude_pid, resolve_session_id

__all__ = [
    'kill_and_copy_resume',
]

_KILL_SCRIPT = """\
import os, signal, time
pid = {claude_pid}
time.sleep(0.5)
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except OSError:
        break
"""


def kill_and_copy_resume(
    *,
    session_id: str | None = None,
    extra_args: Sequence[str] = (),
) -> str:
    """Kill Claude Code and copy the resume command to clipboard.

    Spawns a detached process that sends SIGTERM after 0.5s, then waits
    up to 5s for exit. The resume command is copied to the macOS clipboard
    so the user can Cmd+V + Enter after Claude exits.

    Session ID discovery (first match wins):
        1. ``session_id`` parameter (caller already knows it)
        2. ``$CLAUDE_CODE_SESSION_ID`` env var (set by inject-session-env hook)
        3. ``resolve_session_id()`` from sessions.json

    Args:
        session_id: Session UUID. Discovered automatically if not provided.
        extra_args: Additional flags for the resume command
            (e.g. ``['--model', 'opus']``).

    Returns:
        The resume command string that was copied to clipboard.
    """
    claude_pid = find_claude_pid()

    if session_id is None:
        session_id = os.environ.get('CLAUDE_CODE_SESSION_ID')
    if session_id is None:
        session_id = resolve_session_id(claude_pid, os.getcwd())

    parts = ['claude', '--resume', shlex.quote(session_id), *extra_args]
    resume_cmd = ' '.join(parts)

    subprocess.run(['pbcopy'], input=resume_cmd.encode(), check=False)

    script = _KILL_SCRIPT.format(claude_pid=claude_pid)
    subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    return resume_cmd
