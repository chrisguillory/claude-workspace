"""Pre-shared key authentication and daemon configuration."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path

from filelock import FileLock

__all__ = [
    'DaemonConfig',
    'config_path',
    'generate_key',
    'load_config',
    'save_config',
    'verify_key',
]

CONFIG_DIR = Path.home() / '.claude-workspace' / 'claude-remote-bash'
CONFIG_FILE = CONFIG_DIR / 'config.json'
LOCK_FILE = CONFIG_DIR / 'config.lock'
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
    return CONFIG_FILE


def generate_key() -> str:
    """Generate a cryptographically random 256-bit hex key."""
    return secrets.token_hex(KEY_LENGTH)


def load_config() -> DaemonConfig | None:
    """Load config from disk. Returns None if no config file exists."""
    with FileLock(LOCK_FILE):
        if not CONFIG_FILE.exists():
            return None
        data = json.loads(CONFIG_FILE.read_text())
        return DaemonConfig.from_dict(data)


def save_config(config: DaemonConfig) -> Path:
    """Save config to disk with 0600 permissions. Returns the config file path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0o700 — umask doesn't narrow mkdir mode
    with FileLock(LOCK_FILE):
        CONFIG_FILE.write_text(json.dumps(config.to_dict(), indent=2) + '\n')
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
    return CONFIG_FILE


def verify_key(provided: str, config: DaemonConfig) -> bool:
    """Constant-time comparison of provided key against stored key."""
    return secrets.compare_digest(provided.encode(), config.auth_key.encode())
