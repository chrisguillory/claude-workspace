"""
Base configuration for Claude Session services.

Shared settings and helper functions for all service types (MCP, HTTP).

Data directory: ~/.claude-workspace/claude-session/
  Stores lineage.json, chain-backups/, and deleted/ session backups.
  Auto-migrated from legacy ~/.claude-session-mcp/ on first access.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
from typing import TypeVar, cast

import lazy_object_proxy
import pydantic
import pydantic_settings

T = TypeVar('T', bound='BaseSessionSettings')

logger = logging.getLogger(__name__)

# Canonical data directory for claude-session
DATA_DIR = pathlib.Path.home() / '.claude-workspace' / 'claude-session'

# Legacy data directory (pre-migration)
_LEGACY_DATA_DIR = pathlib.Path.home() / '.claude-session-mcp'

# Track whether migration has been attempted this process
_migration_checked = False


def ensure_data_dir() -> pathlib.Path:
    """Return the data directory, auto-migrating from legacy path if needed.

    On first call per process, if the legacy directory (~/.claude-session-mcp/)
    exists and the new directory (~/.claude-workspace/claude-session/) does not,
    moves contents to the new location.
    """
    global _migration_checked  # noqa: PLW0603

    if not _migration_checked:
        _migration_checked = True
        _migrate_legacy_data_dir()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _migrate_legacy_data_dir() -> None:
    """Move data from ~/.claude-session-mcp/ to ~/.claude-workspace/claude-session/.

    Only runs if the old directory exists and the new one does not.
    """
    if not _LEGACY_DATA_DIR.exists():
        return
    if DATA_DIR.exists():
        return

    logger.warning(
        'Migrating data directory: %s -> %s',
        _LEGACY_DATA_DIR,
        DATA_DIR,
    )

    # Ensure parent exists
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Move the entire directory
    shutil.move(str(_LEGACY_DATA_DIR), str(DATA_DIR))

    logger.warning('Data directory migration complete')


class BaseSessionSettings(pydantic_settings.BaseSettings):
    """Shared configuration across all session services (MCP, HTTP)."""

    model_config = pydantic_settings.SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,  # Fail fast on misconfiguration
        extra='ignore',  # Only read defined fields, ignore other env vars
    )

    # Application metadata
    APP_NAME: str = 'claude-session'
    VERSION: str = '0.1.0'

    # Compression settings (not overrideable at call-time)
    COMPRESSION_LEVEL: int = 3  # 0-9, zstd compression level (3 = balanced)

    @pydantic.field_validator('COMPRESSION_LEVEL')
    @classmethod
    def validate_compression_level(cls, v: int) -> int:
        """Validate compression level is within zstd bounds."""
        if not 0 <= v <= 9:
            raise ValueError('COMPRESSION_LEVEL must be between 0-9')
        return v


def get_settings[T: 'BaseSessionSettings'](settings_class: type[T], env_file: str | None = None) -> T:
    """
    Factory for creating settings with dynamic .env file loading.

    LOAD_ENV_FILE environment variable specifies custom .env file path.
    When unset (production), loads from environment variables only.

    Args:
        settings_class: Settings class to instantiate
        env_file: Optional path to .env file (overrides LOAD_ENV_FILE)

    Returns:
        Settings instance

    Raises:
        FileNotFoundError: If specified .env file doesn't exist
    """
    env_file_path = env_file or os.getenv('LOAD_ENV_FILE')

    if not env_file_path:
        return settings_class()  # No .env file, load from environment only

    resolved_path = pathlib.Path(env_file_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f'Environment file not found: {resolved_path}')

    return settings_class(_env_file=resolved_path)


def lazy_settings[T: 'BaseSessionSettings'](settings_class: type[T]) -> T:
    """Defer settings instantiation until first attribute access.

    Decouples import from validation: modules can import ``settings`` without
    triggering pydantic-settings validation.  Prevents import-time failures
    when environment variables or .env files are missing — validation only
    fires when settings are actually used.
    """
    # Proxy[T] acts as T at runtime - cast to satisfy mypy
    return cast(T, lazy_object_proxy.Proxy(lambda: get_settings(settings_class)))
