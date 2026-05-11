"""Pre-shared key authentication and daemon configuration."""

from __future__ import annotations

import json
import os
import secrets
import stat
from pathlib import Path

from cc_lib.schemas.base import ClosedModel
from filelock import FileLock

__all__ = [
    'CONFIG_FILE',
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


class DaemonConfig(ClosedModel):
    """Daemon configuration loaded from disk."""

    auth_key: str
    name: str
    shell: str
    session_timeout_minutes: int = 1440


def config_path() -> Path:
    """Return the config file path."""
    return CONFIG_FILE


def generate_key() -> str:
    """Generate a cryptographically random 256-bit hex key."""
    return secrets.token_hex(KEY_LENGTH)


def load_config() -> DaemonConfig | None:
    """Load config from disk. Returns None when no config file exists.

    Raises ``json.JSONDecodeError`` or ``pydantic.ValidationError`` when the
    file exists but its contents are malformed.
    """
    with FileLock(LOCK_FILE):
        if not CONFIG_FILE.exists():
            return None
        with open(CONFIG_FILE) as f:
            data = json.load(f)
        return DaemonConfig.model_validate(data)


def save_config(config: DaemonConfig) -> Path:
    """Save config to disk with 0600 permissions. Returns the config file path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CONFIG_DIR, stat.S_IRWXU)  # 0o700 — umask doesn't narrow mkdir mode
    with FileLock(LOCK_FILE):
        CONFIG_FILE.write_text(config.model_dump_json(indent=2) + '\n')
        os.chmod(CONFIG_FILE, stat.S_IRUSR | stat.S_IWUSR)
    return CONFIG_FILE


def verify_key(provided: str, config: DaemonConfig) -> bool:
    """Constant-time comparison of provided key against stored key."""
    return secrets.compare_digest(provided.encode(), config.auth_key.encode())
