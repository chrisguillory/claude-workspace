"""Shared utilities for MCP servers, hooks, and scripts."""

from __future__ import annotations

__all__ = [
    'Timer',
    'encode_project_path',
    'get_claude_config_home_dir',
    'humanize_seconds',
    'load_module_from_path',
    'temporary_module',
]

import importlib.util
import os
import re
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


# Most filesystems cap component names at 255 bytes. Truncate at 200 to
# leave room for the hash suffix.  Mirrors MAX_SANITIZED_LENGTH in
# Claude Code's src/utils/sessionStoragePortable.ts.
MAX_SANITIZED_PATH_LENGTH = 200


def encode_project_path(path: Path | str) -> str:
    """Encode filesystem path for Claude Code's directory naming convention.

    Claude Code stores project data in ``~/.claude/projects/{encoded}/``
    where ``encoded = sanitizePath(cwd)``.  The encoding replaces **every
    non-alphanumeric character** with a hyphen, then truncates long
    results and appends a hash suffix for uniqueness.

    Source of truth: ``sanitizePath()`` in Claude Code's
    ``src/utils/sessionStoragePortable.ts``.

    This encoding is LOSSY and non-reversible.  To recover the original
    path, read the ``cwd`` field from session JSONL records.

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
    name = str(path) if isinstance(path, Path) else path
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', name)
    if len(sanitized) <= MAX_SANITIZED_PATH_LENGTH:
        return sanitized
    # For long paths, truncate and append a hash for uniqueness.
    # Claude Code uses Bun.hash (wyhash) when running under Bun, falling
    # back to djb2Hash under Node.  We use djb2 unconditionally since we
    # are Python, matching the Node/SDK fallback path.
    h = abs(_djb2_hash(name))
    # Base-36 encode (matches JavaScript's ``Math.abs(n).toString(36)``)
    if h == 0:
        hash_str = '0'
    else:
        digits = '0123456789abcdefghijklmnopqrstuvwxyz'
        parts: list[str] = []
        while h > 0:
            parts.append(digits[h % 36])
            h //= 36
        hash_str = ''.join(reversed(parts))
    return f'{sanitized[:MAX_SANITIZED_PATH_LENGTH]}-{hash_str}'


def get_claude_config_home_dir() -> Path:
    """Return Claude Code's config/data root directory.

    Mirrors Claude Code's getClaudeConfigHomeDir() (src/utils/envUtils.ts).
    When CLAUDE_CONFIG_DIR is set, ALL subdirectories (projects/, plans/,
    todos/, tasks/, debug/, session-env/) live under it instead of ~/.claude.

    Resolves custom paths to catch relative paths and typos early — we're
    downstream from Claude Code, so fail-fast on bad input here rather
    than producing cryptic "session not found" errors later.
    """
    config_dir = os.environ.get('CLAUDE_CONFIG_DIR')
    if config_dir:
        return Path(config_dir).resolve()
    return Path.home() / '.claude'


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


def _djb2_hash(s: str) -> int:
    """DJB2 string hash returning a signed 32-bit integer.

    Mirrors ``djb2Hash`` in Claude Code's ``src/utils/hash.ts``.
    Used as the Node.js fallback when ``Bun.hash`` is unavailable.
    """
    h = 0
    for ch in s:
        h = ((h << 5) - h + ord(ch)) & 0xFFFFFFFF
        # Replicate JavaScript's signed 32-bit ``| 0`` semantics
        if h >= 0x80000000:
            h -= 0x100000000
    return h
