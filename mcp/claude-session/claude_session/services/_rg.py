"""Guarded ``rg`` runner shared by the session-file-search services."""

from __future__ import annotations

import subprocess

from cc_lib.system_deps import require_binary

__all__ = [
    'run_rg',
]


def run_rg(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``rg`` for session-file search, raising an actionable error if it is absent.

    Wraps the uniform invocation shared across the session services
    (``check=False``, captured text output); callers parse ``result.stdout``.
    """
    require_binary('rg', needed_for='session file search')
    return subprocess.run(['rg', *args], check=False, capture_output=True, text=True)
