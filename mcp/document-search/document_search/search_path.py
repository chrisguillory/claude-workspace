from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path
from typing import NewType

__all__ = [
    'ResolvedPaths',
    'resolve_filter_paths',
    'resolve_index_paths',
    'resolve_search_paths',
    'to_repo_filter',
]


# Brand returned by the path validators. ``to_repo_filter`` requires this
# type so the type checker rejects raw user input being fed through the
# translator into the repository layer.
ResolvedPaths = NewType('ResolvedPaths', Sequence[str])


def resolve_search_paths(paths: Sequence[str], *, scope_hint: str = 'global scope') -> ResolvedPaths:
    """Validate paths that define a search scope.

    The caller must pick a scope: one or more concrete paths, or ``"**"``
    alone for the entire collection. Empty input is rejected.

    Composes with ``to_repo_filter`` when feeding a repository method or
    ``SearchQuery`` — the translator collapses ``["**"]`` into the empty-list
    sentinel the repository expects. Use this function alone when ``"**"``
    must be preserved (e.g. ``IndexingService.clear_documents``).

    Args:
        paths: User-supplied path inputs. Single strings should be wrapped
            in a list before calling.
        scope_hint: Wording for error messages describing what ``"**"`` means
            for the caller (e.g. ``"global scope"``, ``"entire collection"``).

    Raises:
        ValueError: ``paths`` is empty; ``"**"`` mixed with concrete paths;
            or any individual path contains glob characters or refers to a
            location that does not exist on disk.
    """
    if not paths:
        raise ValueError(f'path cannot be empty. Provide at least one path or "**" for {scope_hint}.')
    if '**' in paths and len(paths) > 1:
        raise ValueError(f'"**" cannot be mixed with other paths. Pass "**" alone for {scope_hint}.')
    return ResolvedPaths([_resolve_search_path(p) for p in paths])


def resolve_filter_paths(paths: Sequence[str]) -> ResolvedPaths:
    """Validate paths that refine a result set.

    Empty input returns ``[]`` (no filter). ``"**"`` is rejected — excluding
    everything is a no-op.

    Raises:
        ValueError: any ``"**"`` in the input; or any individual path
            contains glob characters or refers to a location that does not
            exist on disk.
    """
    if not paths:
        return ResolvedPaths([])
    if any(p == '**' for p in paths):
        raise ValueError('"**" is not supported here. Specify concrete paths.')
    return ResolvedPaths([_resolve_search_path(p) for p in paths])


def resolve_index_paths(paths: Sequence[str]) -> Sequence[Path]:
    """Validate paths for indexing: concrete files or directories only.

    Empty input is rejected. ``"**"`` is rejected. Non-regular filesystem
    entries (sockets, FIFOs, devices) that ``.exists()`` accepts but
    ``IndexingService.index`` cannot consume are also rejected.

    Returns ``Path`` objects so the caller doesn't repeat ``[Path(p) for p in ...]``.

    Raises:
        ValueError: ``paths`` is empty; any individual path fails per-element
            validation; ``"**"`` appears in the input; or a resolved path
            exists but is not a regular file or directory.
    """
    if not paths:
        raise ValueError('path cannot be empty. Provide at least one path.')
    resolved = [Path(p) for p in resolve_filter_paths(paths)]
    for rp in resolved:
        if not rp.is_file() and not rp.is_dir():
            raise ValueError(f'Path is not a file or directory: {rp}')
    return resolved


def to_repo_filter(value: ResolvedPaths) -> Sequence[str]:
    """Translate the user-facing ``"**"`` sentinel to the repository's empty-list form.

    Pure translator — does not validate. Performs the boundary swap so the
    repository layer never sees ``"**"``.

    Compose at the boundary::

        source_prefixes = to_repo_filter(
            resolve_search_paths(paths, scope_hint='global scope'),
        )

    Not needed after ``resolve_filter_paths`` or ``resolve_index_paths``
    (their outputs can never contain ``"**"``).
    """
    return [] if '**' in value else list(value)


def _resolve_search_path(path: str) -> str:
    """Validate and resolve one path string, preserving ``"**"``.

    Rejects glob characters and paths that do not exist on disk. Returns
    the resolved absolute path, or the ``"**"`` sentinel unchanged.
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
