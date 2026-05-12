"""Filesystem paths for claude-remote-bash user-data state."""

from __future__ import annotations

from pathlib import Path

from cc_lib.utils import get_claude_workspace_config_home_dir

__all__ = [
    'CLIENT_CONFIG',
    'DAEMON_CONFIG',
    'DAEMON_CONFIG_LOCK',
    'DATA_DIR',
    'HOSTS_CACHE',
]

DATA_DIR: Path = get_claude_workspace_config_home_dir() / 'mcp' / 'claude-remote-bash'
"""Per-user data directory holding both daemon and client state."""

DAEMON_CONFIG: Path = DATA_DIR / 'daemon_config.json'
"""Daemon's PSK, alias, and shell."""

DAEMON_CONFIG_LOCK: Path = DATA_DIR / 'daemon_config.lock'
"""filelock guard for concurrent daemon-config writes."""

CLIENT_CONFIG: Path = DATA_DIR / 'client_config.json'
"""User's named host groups."""

HOSTS_CACHE: Path = DATA_DIR / 'hosts-cache.json'
"""mDNS browse-result cache."""
