"""Pre-shared key authentication and daemon configuration."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from filelock import FileLock

from claude_remote_bash.paths import DAEMON_CONFIG, DAEMON_CONFIG_LOCK, DATA_DIR

__all__ = [
    'DaemonConfig',
    'config_path',
    'generate_key',
    'load_config',
    'save_config',
    'verify_key',
]

KEY_LENGTH = 32  # 256-bit key


class DaemonConfig:
    """Daemon configuration loaded from disk.

    Not a Pydantic model — this is local configuration, not protocol data.
    Mutable because the daemon may update fields at runtime.
    """

    def __init__(
        self,
        *,
        name: str = '',
        auth_key: str = '',
        shell: str = '/bin/zsh',
        session_timeout_minutes: int = 1440,
    ) -> None:
        self.name = name
        self.auth_key = auth_key
        self.shell = shell
        self.session_timeout_minutes = session_timeout_minutes

    def to_dict(self) -> Mapping[str, object]:
        return {
            'name': self.name,
            'auth_key': self.auth_key,
            'shell': self.shell,
            'session_timeout_minutes': self.session_timeout_minutes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DaemonConfig:
        return cls(
            name=str(data.get('name', '')),
            auth_key=str(data.get('auth_key', '')),
            shell=str(data.get('shell', '/bin/zsh')),
            session_timeout_minutes=int(str(data.get('session_timeout_minutes', 1440))),
        )


def config_path() -> Path:
    """Return the config file path."""
    return DAEMON_CONFIG


def generate_key() -> str:
    """Generate a cryptographically random 256-bit hex key."""
    return secrets.token_hex(KEY_LENGTH)


def load_config() -> DaemonConfig | None:
    """Load config from disk. Returns None if no config file exists."""
    with FileLock(DAEMON_CONFIG_LOCK):
        if not DAEMON_CONFIG.exists():
            return None
        data = json.loads(DAEMON_CONFIG.read_text())
        return DaemonConfig.from_dict(data)


def save_config(config: DaemonConfig) -> Path:
    """Save config to disk with 0600 permissions. Returns the config file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_DIR, stat.S_IRWXU)  # 0o700 — umask doesn't narrow mkdir mode
    with FileLock(DAEMON_CONFIG_LOCK):
        DAEMON_CONFIG.write_text(json.dumps(config.to_dict(), indent=2) + '\n')
        os.chmod(DAEMON_CONFIG, stat.S_IRUSR | stat.S_IWUSR)
    return DAEMON_CONFIG


def verify_key(provided: str, config: DaemonConfig) -> bool:
    """Constant-time comparison of provided key against stored key."""
    return secrets.compare_digest(provided.encode(), config.auth_key.encode())
