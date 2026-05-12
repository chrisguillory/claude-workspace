from __future__ import annotations

__all__ = [
    'claude_binary_name',
    'get_cc_env_var',
    'get_cc_setting',
    'is_env_truthy',
]

import json
import os
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path

from cc_lib.utils import get_claude_config_home_dir


def is_env_truthy(value: str | None) -> bool:
    """Mirror Claude Code's ``yH()`` env-truthiness check.

    Truthy values: ``'1'``, ``'true'``, ``'yes'`` (case-insensitive).
    Anything else (``'0'``, ``'false'``, empty string, ``None``) is falsy.
    """
    return value is not None and value.lower() in ('1', 'true', 'yes')


def claude_binary_name() -> str:
    """Return the binary name the user invokes to launch Claude Code.

    Reads ``CLAUDE_BINARY_NAME``. Defaults to ``'claude'``.
    """
    return get_cc_env_var('CLAUDE_BINARY_NAME') or 'claude'


def get_cc_env_var(key: str) -> str | None:
    """Read a Claude Code env var, resolving from shell env then settings.json chain.

    Precedence (highest first): existing os.environ values, project local
    settings, project settings, user local settings, user settings. Shell-set
    values are checked first and never overridden.

    Use this for any env var defined or used by Claude Code, cc_lib, or
    workspace scripts (CC_*, CLAUDE_*, ANTHROPIC_*) so that bare-shell
    invocations honor values set in settings.json. Don't use it for system
    env vars (PATH, USER, SHELL, etc.) — those won't be in settings.json
    and a plain os.environ.get is more direct.

    Returns None if unset everywhere.
    """
    if (value := os.environ.get(key)) is not None:
        return value
    return _merged_env_settings().get(key)


def get_cc_setting(key: str) -> object | None:
    """Read a top-level Claude Code settings.json key (NOT under env block).

    Use for top-level config keys like ``autoMemoryEnabled``,
    ``autoCompactEnabled``, ``permissions``, etc. — distinct from
    ``get_cc_env_var`` which reads the ``env`` sub-block.

    Precedence matches ``get_cc_env_var`` for the settings.json chain:
    user > user local > project > project local (later files override).

    Returns the raw value (any JSON type — caller must narrow with
    ``isinstance`` or equality) or None if unset everywhere.
    """
    return _merged_top_level_settings().get(key)


@cache
def _merged_env_settings() -> Mapping[str, str]:
    """Merge env blocks from settings.json files. Cached for process lifetime."""
    merged: dict[str, str] = {}
    for data in _settings_data():
        env_block = data.get('env')
        if not isinstance(env_block, dict):
            continue
        for k, v in env_block.items():
            if isinstance(k, str):
                merged[k] = str(v)
    return merged


@cache
def _merged_top_level_settings() -> Mapping[str, object]:
    """Merge top-level keys from settings.json files. Cached for process lifetime."""
    merged: dict[str, object] = {}
    for data in _settings_data():
        for k, v in data.items():
            if k == 'env':
                continue  # handled by _merged_env_settings
            merged[k] = v
    return merged


def _settings_data() -> Sequence[Mapping[str, object]]:
    """Load and parse all settings.json files in precedence order (low → high)."""
    config_home = get_claude_config_home_dir()
    cwd = Path.cwd()
    paths = (
        config_home / 'settings.json',
        config_home / 'settings.local.json',
        cwd / '.claude' / 'settings.json',
        cwd / '.claude' / 'settings.local.json',
    )
    out: list[Mapping[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        out.append(json.loads(path.read_text()))
    return out
