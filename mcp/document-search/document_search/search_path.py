from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path

__all__ = [
    'resolve_filter_paths',
    'resolve_index_paths',
    'resolve_search_paths',
    'to_repo_filter',
]


def resolve_search_paths(paths: Sequence[str], *, scope_hint: str = 'global scope') -> Sequence[str]:
    """Validate paths that define a *search scope* — must specify what to look at.

    Used for ``--path`` style inputs (search/list/info/clear). The caller
    must pick a scope: one or more concrete paths, or ``"**"`` alone for
    the entire collection.

    Most callers feeding the result to a repository method or ``SearchQuery``
    should compose with ``to_repo_filter`` to collapse ``["**"]`` into the
    repository's empty-list sentinel ``[]``. Use this function directly only
    when ``"**"`` should be preserved (e.g. propagating to
    ``IndexingService.clear_documents``).

    Args:
        paths: User-supplied path inputs. Already shape-normalized by the
            caller (single strings should be wrapped in a list before calling).
        scope_hint: Wording used in error messages to describe what ``"**"``
            means for the calling tool (e.g. ``"global scope"`` for search,
            ``"entire collection"`` for clear).

    Raises:
        ValueError: ``paths`` is empty; ``"**"`` mixed with concrete paths;
            or any individual path contains glob characters or refers to a
            location that does not exist on disk.
    """
    if not paths:
        raise ValueError(f'path cannot be empty. Provide at least one path or "**" for {scope_hint}.')
    if '**' in paths and len(paths) > 1:
        raise ValueError(f'"**" cannot be mixed with other paths. Pass "**" alone for {scope_hint}.')
    return [_resolve_search_path(p) for p in paths]


def resolve_filter_paths(paths: Sequence[str]) -> Sequence[str]:
    """Validate paths that *refine* a result set — empty means "no filter".

    Used for ``--exclude`` style inputs and any other optional path-based
    refinement. Empty input is the natural identity (no filter); ``"**"``
    is meaningless here (excluding everything is a no-op) and rejected.

    Raises:
        ValueError: any ``"**"`` in the input; or any individual path
            contains glob characters or refers to a location that does not
            exist on disk.
    """
    if not paths:
        return []
    if any(p == '**' for p in paths):
        raise ValueError('"**" is not supported here. Specify concrete paths.')
    return [_resolve_search_path(p) for p in paths]


def resolve_index_paths(paths: Sequence[str]) -> Sequence[Path]:
    """Validate paths for indexing: concrete files or directories only.

    Wraps ``resolve_filter_paths`` and additionally rejects empty input
    (indexing nothing is meaningless) and non-regular filesystem entries
    (sockets, FIFOs, devices) that ``.exists()`` accepts but
    ``IndexingService.index`` cannot consume.

    Returns a list of resolved ``Path`` objects so the caller doesn't have
    to redo the ``[Path(p) for p in ...]`` wrap.

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


def to_repo_filter(value: Sequence[str]) -> Sequence[str]:
    """Translate the user-facing ``"**"`` sentinel to the repository's empty-list form.

    Pure translator. Validation belongs to ``resolve_search_paths``; this
    function trusts that its input has already been validated and just
    performs the boundary swap so the repository layer never sees ``"**"``.

    Compose at the boundary::

        source_prefixes = to_repo_filter(
            resolve_search_paths(paths, scope_hint='global scope'),
        )

    Use when feeding a repository method (``get_content_stats``,
    ``list_indexed_files``) or constructing a ``SearchQuery``. Not needed
    for ``resolve_filter_paths`` results (they can never contain ``"**"``)
    or ``resolve_index_paths`` (same reason).
    """
    return [] if '**' in value else list(value)


def _resolve_search_path(path: str) -> str:
    """Validate and resolve one user-supplied path, preserving ``"**"``.

    Shared per-element helper. Not part of the public interface — callers
    pick one of the plural functions above based on whether the path list
    defines a scope, refines results, or targets indexing.
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
