"""Static identity of a workspace project (MCP server, CLI, daemon).

Companion to ``cc_lib.mcp.McpServerInfo``: ``Project`` is what the package
declares about itself before any process starts; ``McpServerInfo`` is what
the running process registers (pid, session, socket, capabilities).
"""

from __future__ import annotations

__all__ = [
    'Project',
]

import tomllib
from pathlib import Path

from cc_lib.schemas.base import ClosedModel


class Project(ClosedModel):
    """A workspace project's static identity."""

    name: str
    """Kebab-case canonical identifier, read from ``[project] name`` in pyproject.toml."""

    @classmethod
    def from_pyproject(cls, module_file: str) -> Project:
        """Build from the nearest ``pyproject.toml`` at or above ``module_file``.

        Call with ``__file__`` from a package ``__init__.py``. Under
        ``uv tool install --editable`` the package imports from the source tree,
        so its ``pyproject.toml`` is found by walking up from the module —
        making ``[project] name`` the single source of the project's identity.
        """
        pyproject = _find_pyproject(Path(module_file))
        return cls(name=tomllib.loads(pyproject.read_text())['project']['name'])


def _find_pyproject(module_file: Path) -> Path:
    """Return the nearest ``pyproject.toml`` at or above ``module_file``'s directory."""
    start = module_file if module_file.is_dir() else module_file.parent
    for directory in (start, *start.parents):
        candidate = directory / 'pyproject.toml'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'no pyproject.toml at or above {module_file}')
