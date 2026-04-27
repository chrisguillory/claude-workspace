from __future__ import annotations

__all__ = [
    'get_cc_env_var',
]

import json
import os
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from cc_lib.utils import get_claude_config_home_dir


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
    return _merged_settings().get(key)


@cache
def _merged_settings() -> Mapping[str, str]:
    """Merge env blocks from settings.json files. Cached for process lifetime."""
    config_home = get_claude_config_home_dir()
    cwd = Path.cwd()
    paths = (
        config_home / 'settings.json',
        config_home / 'settings.local.json',
        cwd / '.claude' / 'settings.json',
        cwd / '.claude' / 'settings.local.json',
    )

    merged: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        env_block = data.get('env')
        if not isinstance(env_block, dict):
            continue
        for k, v in env_block.items():
            if isinstance(k, str):
                merged[k] = str(v)
    return merged
