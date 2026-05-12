"""Configure test environment for mypy plugin tests.

The pydantic_replace plugin is referenced by module path
(``plugins.pydantic_replace``) in YAML test configs. When pytest-mypy-plugins
spawns mypy as a subprocess, the plugins package must be importable —
PYTHONPATH is the standard mechanism for this.
"""

from __future__ import annotations

__all__ = [
    'pytest_configure',
]

import os
from pathlib import Path


def pytest_configure() -> None:
    """Add repo root to PYTHONPATH for mypy subprocess plugin loading.

    Complements pythonpath=["."] in pyproject.toml which only affects
    the pytest process itself, not the mypy subprocess.
    """
    repo_root = str(Path(__file__).resolve().parent)
    os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [repo_root, os.environ.get('PYTHONPATH', '')]))

    # Set to '' (not pop) — get_cc_env_var falls back to ~/.claude/settings.json
    # when a CC_* var is unset, which would otherwise read user config in tests.
    os.environ['CC_STRICT_MODEL_EXTRA_FORBID'] = ''
    os.environ['CC_OPEN_MODEL_EXTRA_FORBID'] = ''
    os.environ['CC_LITERAL_RELAX'] = ''
