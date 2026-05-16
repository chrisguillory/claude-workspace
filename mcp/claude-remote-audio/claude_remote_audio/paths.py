from __future__ import annotations

from pathlib import Path

from cc_lib.utils import get_claude_workspace_config_home_dir

__all__ = [
    'DATA_DIR',
    'LOGS_DIR',
]

DATA_DIR: Path = get_claude_workspace_config_home_dir() / 'mcp' / 'claude-remote-audio'
"""Per-user data directory for persistent state."""

LOGS_DIR: Path = DATA_DIR / 'logs'
"""Per-apply-run log files (``apply-YYYYMMDD-HHMMSS.log``, DEBUG-level trace)."""
