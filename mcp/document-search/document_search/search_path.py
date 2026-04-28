from __future__ import annotations

from pathlib import Path

__all__ = [
    'resolve_search_path',
]

_GLOB_CHARS = frozenset('*?[]')


def resolve_search_path(path: str | None) -> str:
    """Resolve and validate a user-supplied search path.

    Centralizes the ``--path`` / ``path`` contract for the CLI and MCP server.
    Fail-fast on globs and missing paths so callers see an actionable error
    instead of qdrant silently filtering out every result.

    Args:
        path: Raw input. ``None`` defaults to the current working directory.
            ``"**"`` is the explicit global-scope sentinel.

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
    if path is None:
        return str(Path.cwd())

    if any(c in _GLOB_CHARS for c in path):
        raise ValueError(f'Glob characters not supported in path filter: {path!r}. Use "**" for global scope.')

    expanded = Path(path).expanduser()
    if not expanded.exists():
        raise ValueError(f'Path does not exist: {path!r}.')

    return str(expanded.resolve())
