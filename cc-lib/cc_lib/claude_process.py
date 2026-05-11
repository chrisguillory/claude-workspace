from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence

from cc_lib.claude_context import ClaudeContext

__all__ = [
    'kill_and_copy_resume',
]


def kill_and_copy_resume(
    claude_context: ClaudeContext,
    *,
    extra_args: Sequence[str] = (),
) -> str:
    """Kill Claude Code and copy `claude --resume <session-id>` to clipboard.

    SIGTERM fires 0.5s after this returns (detached subprocess), giving the
    caller time to print success messages before Claude exits.
    """
    parts = [
        'claude',
        '--resume',
        shlex.quote(claude_context.session_id),
        *(shlex.quote(a) for a in extra_args),
    ]
    resume_cmd = ' '.join(parts)

    subprocess.run(['pbcopy'], input=resume_cmd.encode(), check=False)

    subprocess.Popen(
        [sys.executable, '-c', _KILL_SCRIPT, str(claude_context.claude_pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    return resume_cmd


_KILL_SCRIPT = """\
import os, signal, sys, time
pid = int(sys.argv[1])
time.sleep(0.5)
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except OSError:
        break
"""
