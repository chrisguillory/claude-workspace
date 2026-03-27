"""Shared utilities for MCP servers, hooks, and scripts."""

from __future__ import annotations

__all__ = [
    'Timer',
    'encode_project_path',
    'humanize_seconds',
    'load_module_from_path',
    'temporary_module',
]

import importlib.util
import sys
import time
import unittest.mock
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType


class Timer:
    """Simple stopwatch-style timer for measuring elapsed time."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.perf_counter() - self._start

    def elapsed_ms(self) -> int:
        """Return elapsed time in milliseconds."""
        return int(self.elapsed() * 1000)


def encode_project_path(path: Path | str) -> str:
    """Encode filesystem path for Claude Code's directory naming convention.

    Claude Code stores project data in ~/.claude/projects/{encoded_path}/ where
    the path is encoded by replacing certain characters with hyphens.

    This encoding is LOSSY and non-reversible (5 different chars -> 1 char).
    To recover the original path, read the 'cwd' field from session records.

    Args:
        path: Filesystem path to encode (e.g., /Users/chris/My Project)

    Returns:
        Encoded string for use as directory name in ~/.claude/projects/

    Examples:
        >>> encode_project_path("/Users/chris/project")
        '-Users-chris-project'

        >>> encode_project_path("/Users/chris/Mobile Documents/com~apple~CloudDocs")
        '-Users-chris-Mobile-Documents-com-apple-CloudDocs'

        >>> encode_project_path("/Users/chris/my_project.app")
        '-Users-chris-my-project-app'
    """
    result = str(path) if isinstance(path, Path) else path
    # Claude Code encodes paths by replacing these characters with hyphens:
    for char in ['/', '.', ' ', '~', '_']:
        result = result.replace(char, '-')
    return result


def humanize_seconds(seconds: float) -> str:
    """Convert seconds to human-readable duration with abbreviated units.

    Uses terse but complete format following Unix conventions:
    - 45 sec
    - 1.5 min
    - 2.5 hr
    - 3d

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string with abbreviated unit
    """
    intervals = [
        ('d', 86400),  # days
        ('hr', 3600),  # hours
        ('min', 60),  # minutes
        ('sec', 1),  # seconds
    ]

    for unit, count in intervals:
        if seconds >= count:
            value = seconds / count
            value_str = f'{value:.1f}'.rstrip('0').rstrip('.')
            return f'{value_str} {unit}'

    return '0 sec'


def load_module_from_path(path: str | Path, module_name: str | None = None) -> ModuleType:
    """Load a Python module from a filesystem path.

    Bypasses normal import machinery — useful for files with non-identifier
    names (e.g. ``approve-compound-bash.py``).

    Does NOT register the module in ``sys.modules``. Use ``temporary_module()``
    for automatic registration and cleanup.

    Note:
        Registration happens after ``exec_module``. If the loaded module
        imports itself during execution, use the importlib primitives directly
        with pre-registration.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'module file not found: {path}')
    if module_name is None:
        module_name = path.stem.replace('-', '_')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'failed to create module spec for {path}')
    loader = spec.loader  # bind for type narrowing
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@contextmanager
def temporary_module(
    path: str | Path,
    module_name: str | None = None,
) -> Iterator[ModuleType]:
    """Load a module and temporarily register it in ``sys.modules``.

    Restores prior ``sys.modules`` state on exit via ``patch.dict``.
    Useful in test fixtures where submodule imports need the parent registered.
    """
    mod = load_module_from_path(path, module_name)
    with unittest.mock.patch.dict(sys.modules, {mod.__name__: mod}):
        yield mod
