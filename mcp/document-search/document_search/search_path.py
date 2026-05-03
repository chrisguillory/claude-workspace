from __future__ import annotations

from collections.abc import Set
from pathlib import Path

__all__ = [
    'resolve_search_path',
]


def resolve_search_path(path: str) -> str:
    """Validate and resolve a user-supplied search path.

    Pure validator. Defaults (e.g. cwd when no path is supplied) are a UX
    policy decision and belong to the caller; this function deals only
    with concrete strings.

    Args:
        path: Raw input string. ``"**"`` is the explicit global-scope sentinel.

    Returns:
        Resolved absolute path string for prefix matching, or the literal
        ``"**"`` for global scope. Callers that disallow global scope
        (e.g. indexing) should reject ``"**"`` themselves.

    Raises:
        ValueError: Path contains glob characters other than the ``"**"``
            sentinel, or refers to a location that does not exist on disk.
    """
    if path == '**':
        return '**'

    if any(c in GLOB_CHARS for c in path):
        raise ValueError(f'Glob characters not supported in path filter: {path!r}. Use "**" for global scope.')

    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise ValueError(f'Path does not exist: {path!r}.')

    return str(expanded.resolve())


GLOB_CHARS: Set[str] = {
    '*',
    '?',
    '[',
    ']',
}
