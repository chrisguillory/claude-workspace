"""Presence guards for external system binaries (require_binary)."""

from __future__ import annotations

import shutil

from cc_lib.exceptions import MissingSystemDependency

__all__ = [
    'require_binary',
]

_DEFAULT_INSTALL_HINT = 'Run ./dotfiles/install.sh from the repo root to install the project toolchain.'


def require_binary(binary: str, *, needed_for: str, install_hint: str = _DEFAULT_INSTALL_HINT) -> str:
    """Return the resolved path to ``binary``, or raise ``MissingSystemDependency``.

    Guards a shell-out at the boundary. ``needed_for`` names the feature for the
    error message; ``install_hint`` overrides the default remedy for binaries
    installed by another mechanism (a venv tool, ``uv tool install``).
    """
    path = shutil.which(binary)
    if path is None:
        raise MissingSystemDependency(binary, needed_for=needed_for, install_hint=install_hint)
    return path
