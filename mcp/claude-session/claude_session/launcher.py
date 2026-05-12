"""Claude Code launcher utility.

Provides functionality to launch Claude Code with a specific session.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

from cc_lib.settings_env import claude_binary_name

__all__ = [
    'launch_claude_with_session',
]


def launch_claude_with_session(session_id: str, extra_args: Sequence[str]) -> None:
    """Launch Claude Code with --resume, replacing current process.

    Uses os.execvp() for clean process handoff - the current process
    is replaced by Claude Code, so this function never returns.

    Args:
        session_id: Session ID to resume
        extra_args: Additional arguments to pass to claude CLI (e.g., --chrome)

    Raises:
        RuntimeError: If the resolved claude binary is not found in PATH
    """
    binary = claude_binary_name()
    if not shutil.which(binary):
        raise RuntimeError(f'{binary!r} not found in PATH.\nInstall from: https://claude.ai/code')

    cmd = [binary, '--resume', session_id]
    if extra_args:
        cmd.extend(extra_args)

    os.execvp(binary, cmd)
