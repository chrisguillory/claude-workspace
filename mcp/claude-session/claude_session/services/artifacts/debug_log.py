"""
Debug log artifact handling.

Collects and writes per-session debug logs stored at
~/.claude/debug/<session-id>.txt.
"""

from __future__ import annotations

from pathlib import Path

from cc_lib.utils import get_claude_config_home_dir

__all__ = [
    'collect_debug_log',
    'write_debug_log',
]


def collect_debug_log(session_id: str) -> str | None:
    """Read debug/<session-id>.txt if it exists.

    Args:
        session_id: The session UUID

    Returns:
        The debug log content, or None if no debug log exists.
    """
    path = get_claude_config_home_dir() / 'debug' / f'{session_id}.txt'
    if path.exists():
        return path.read_text()
    return None


def write_debug_log(session_id: str, content: str) -> Path:
    """Write debug/<session-id>.txt.

    Args:
        session_id: The session UUID
        content: The debug log content

    Returns:
        Path to the written file.
    """
    path = get_claude_config_home_dir() / 'debug' / f'{session_id}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
