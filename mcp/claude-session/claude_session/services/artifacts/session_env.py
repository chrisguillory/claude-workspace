"""
Session-env artifact handling.

Collects and writes per-session environment files stored at
~/.claude/session-env/<session-id>/.

These files are hook-generated scripts (e.g., sessionstart-hook-1.sh)
that inject environment variables into the session.
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
    """Collect all files from session-env/<session-id>/.

    Args:
        session_id: Session ID to collect

    Returns:
        Mapping of filename to content. Empty dict if directory doesn't exist.
    """
    session_env_path = get_session_env_dir() / session_id
    if not session_env_path.exists():
        return {}

    result: dict[str, str] = {}
    for path in sorted(session_env_path.iterdir()):
        if path.is_file():
            result[path.name] = path.read_text(encoding='utf-8')
    return result


def write_session_env(session_id: str, files: Mapping[str, str]) -> int:
    """Write session-env files for a session.

    Args:
        session_id: Session ID
        files: Mapping of filename to content

    Returns:
        Number of files written.
    """
    if not files:
        return 0
    session_env_path = get_session_env_dir() / session_id
    session_env_path.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (session_env_path / filename).write_text(content, encoding='utf-8')
    return len(files)


def create_session_env_dir(session_id: str) -> Path:
    """Create empty session-env directory for a new session.

    Args:
        session_id: New session ID

    Returns:
        Path to created (or existing) directory
    """
    session_env_path = get_session_env_dir() / session_id
    session_env_path.mkdir(parents=True, exist_ok=True)
    return session_env_path
