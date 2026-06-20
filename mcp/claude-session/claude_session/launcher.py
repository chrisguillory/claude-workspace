"""Claude Code launcher utility.

Provides functionality to launch Claude Code with a specific session.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from cc_lib.settings_env import claude_binary_name
from cc_lib.system_deps import require_binary

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
        MissingSystemDependency: If the resolved claude binary is not on PATH
    """
    binary = claude_binary_name()
    require_binary(binary, needed_for='resuming a session', install_hint='Install from: https://claude.ai/code')

    cmd = [binary, '--resume', session_id]
    if extra_args:
        cmd.extend(extra_args)

    os.execvp(binary, cmd)
