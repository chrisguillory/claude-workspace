"""
Session-env artifact handling for archive/restore operations.

Path pattern: ~/.claude/session-env/<session-id>/

Session-env contains small text files written by hooks (e.g., inject-session-env.py).
Files are collected as a flat {filename: content} mapping and restored verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from cc_lib.utils import get_claude_config_home_dir

__all__ = [
    'collect_session_env',
    'create_session_env_dir',
    'get_session_env_dir',
    'write_session_env',
]


def get_session_env_dir() -> Path:
    """Return session-env directory, respecting CLAUDE_CONFIG_DIR."""
    return get_claude_config_home_dir() / 'session-env'


def collect_session_env(session_id: str) -> Mapping[str, str]:
    """
    Collect all files from the session-env directory.

    Reads all files in ~/.claude/session-env/<session-id>/ and returns
    them as a mapping of filename to content. Only collects flat files
    (no recursive subdirectory traversal).

    If the session-env directory doesn't exist or is empty, returns
    an empty mapping. This is NOT an error condition.

    Args:
        session_id: Session ID to collect files for

    Returns:
        Mapping of filename -> text content
    """
    session_env_path = get_session_env_dir() / session_id

    if not session_env_path.exists():
        return {}

    results: dict[str, str] = {}
    for path in sorted(session_env_path.iterdir()):
        if path.is_file():
            results[path.name] = path.read_text(encoding='utf-8')

    return results


def write_session_env(session_id: str, files: Mapping[str, str]) -> int:
    """
    Write session-env files for a session.

    Creates the session-env directory and writes each file.

    Args:
        session_id: Target session ID
        files: Mapping of filename -> text content

    Returns:
        Number of files written
    """
    if not files:
        return 0

    session_env_path = get_session_env_dir() / session_id
    session_env_path.mkdir(parents=True, exist_ok=True)

    for filename, content in files.items():
        (session_env_path / filename).write_text(content, encoding='utf-8')

    return len(files)


def create_session_env_dir(session_id: str) -> Path:
    """
    Create empty session-env directory for a new session.

    Creates the directory structure to match what Claude Code expects.
    Uses mkdir with exist_ok=True for idempotency.

    Args:
        session_id: New session ID

    Returns:
        Path to created (or existing) directory
    """
    session_env_path = get_session_env_dir() / session_id
    session_env_path.mkdir(parents=True, exist_ok=True)
    return session_env_path
