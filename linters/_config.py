"""Shared config reader for per-file-ignores across custom linters.

Reads ``[tool.<linter-name>.per-file-ignores]`` from pyproject.toml,
following ruff's config discovery pattern: walk up from each file being
checked, skip pyproject.toml files without the tool's section.

Tool names use hyphens in pyproject.toml (``strict-typing-linter``)
matching pre-commit hook IDs, while filenames use underscores
(``strict_typing_linter.py``) matching Python module conventions.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence, Set
from functools import lru_cache
from pathlib import Path, PurePosixPath


def find_config(filepath: Path, tool_name: str) -> tuple[Path, Path] | None:
    """Walk up from filepath to find pyproject.toml with the tool's section.

    Skips pyproject.toml files that parse but lack ``[tool.<tool_name>]``.
    Stops at filesystem root.

    Returns:
        (config_path, project_root) tuple, or None if no config found.
    """
    current = filepath.resolve().parent
    for directory in [current, *current.parents]:
        candidate = directory / 'pyproject.toml'
        if not candidate.is_file():
            continue
        with candidate.open('rb') as f:
            data = tomllib.load(f)
        if tool_name in data.get('tool', {}):
            return candidate, directory
    return None


def load_per_file_ignores(
    tool_name: str,
    config_path: Path,
) -> Mapping[str, Sequence[str]]:
    """Read per-file-ignores from a pyproject.toml for a given tool.

    Returns:
        Mapping from glob pattern to list of violation codes to ignore.
        Empty mapping if no per-file-ignores section.
    """
    return _load_per_file_ignores_cached(tool_name, config_path.resolve())


@lru_cache(maxsize=32)
def _load_per_file_ignores_cached(
    tool_name: str,
    config_path: Path,
) -> Mapping[str, Sequence[str]]:
    with config_path.open('rb') as f:
        data = tomllib.load(f)
    raw = data.get('tool', {}).get(tool_name, {}).get('per-file-ignores', {})
    for pattern, codes in raw.items():
        if not isinstance(codes, list):
            msg = f'per-file-ignores value for {pattern!r} must be a list, got {type(codes).__name__}: use ["{codes}"] instead of "{codes}"'
            raise TypeError(msg)
    result: Mapping[str, Sequence[str]] = raw
    return result


def get_per_file_ignored_codes(
    filepath: Path,
    per_file_ignores: Mapping[str, Sequence[str]],
    project_root: Path,
) -> Set[str]:
    """Resolve which violation codes to ignore for a specific file.

    Relativizes filepath to project_root before matching against globs.
    Returns empty set for files outside the project root.
    """
    if not per_file_ignores:
        return set()

    resolved = filepath.resolve()
    root = project_root.resolve()
    if not resolved.is_relative_to(root):
        return set()
    relative = PurePosixPath(resolved.relative_to(root))

    ignored: set[str] = set()
    for glob_pattern, codes in per_file_ignores.items():
        if relative.full_match(glob_pattern):
            ignored.update(codes)
    return ignored
