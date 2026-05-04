from __future__ import annotations

from collections.abc import Sequence, Set
from pathlib import Path
from typing import overload

__all__ = [
    'resolve_index_paths',
    'resolve_search_path',
    'resolve_search_paths',
    'to_repo_filter',
]


def resolve_index_paths(paths: Sequence[str]) -> Sequence[Path]:
    """Validate paths for indexing: concrete files or directories only.

    Wraps ``resolve_search_paths(allow_global=False)`` and additionally
    rejects non-regular filesystem entries (sockets, FIFOs, devices) that
    ``.exists()`` accepts but ``IndexingService.index`` cannot consume.

    Returns a list of resolved ``Path`` objects so the caller doesn't
    have to redo the ``[Path(p) for p in ...]`` wrap.

    Raises:
        ValueError: Any individual path fails ``resolve_search_path``,
            ``"**"`` appears in the input, or a resolved path exists but
            is not a regular file or directory.
    """
    resolved = [Path(p) for p in resolve_search_paths(paths, allow_global=False)]
    for rp in resolved:
        if not rp.is_file() and not rp.is_dir():
            raise ValueError(f'Path is not a file or directory: {rp}')
    return resolved


def resolve_search_path(path: str) -> str:
    """Validate and resolve a user-supplied search path, preserving ``"**"``.

    Pure validator. Defaults (e.g. cwd when no path is supplied) are a UX
    policy decision and belong to the caller; this function deals only
    with concrete strings.

    Most callers feeding the result to a repository method should use
    ``to_repo_filter`` instead — it performs the same validation AND
    translates ``"**"`` into the repository's empty-value sentinel.
    Use this lower-level function only when ``"**"`` should be preserved
    (e.g. propagating to ``IndexingService.clear_documents``, or per-element
    validation in indexing where ``"**"`` must be rejected explicitly).

    Args:
        path: Raw input string. ``"**"`` is the explicit global-scope sentinel.

    Returns:
        Resolved absolute path string for prefix matching, or the literal
        ``"**"`` for global scope.

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


def resolve_search_paths(
    paths: Sequence[str], *, scope_hint: str = 'global scope', allow_global: bool = True
) -> Sequence[str]:
    """Validate and resolve a list of user-supplied search paths, preserving ``"**"``.

    Enforces list-level invariants (non-empty; ``"**"`` policy per ``allow_global``)
    and per-element validity via ``resolve_search_path``. Returns a list of
    resolved absolute paths, or ``["**"]`` when global scope was requested
    and is allowed.

    Most callers feeding the result to a repository method or ``SearchQuery``
    should use ``to_repo_filter`` instead — it performs the same validation
    AND collapses ``["**"]`` into the repository's empty-list sentinel ``[]``.
    Use this lower-level function only when ``"**"`` should be preserved
    (e.g. propagating to ``IndexingService.clear_documents``) or when
    ``"**"`` should be rejected entirely (``allow_global=False`` for indexing).

    Args:
        paths: User-supplied path inputs. Already shape-normalized by the
            caller (single strings should be wrapped in a list before calling).
        scope_hint: Wording used in error messages to describe what ``"**"``
            means for the calling tool (e.g. ``"global scope"`` for search,
            ``"entire collection"`` for clear). Ignored when ``allow_global``
            is False.
        allow_global: When True (default), ``"**"`` alone selects global scope.
            When False, ``"**"`` is rejected entirely (e.g. for indexing,
            which requires concrete paths).

    Raises:
        ValueError: Empty list; ``"**"`` mixed with concrete paths (when
            ``allow_global``); any ``"**"`` (when not ``allow_global``); or
            any individual path fails ``resolve_search_path`` validation.
    """
    if not paths:
        suffix = f' or "**" for {scope_hint}' if allow_global else ''
        raise ValueError(f'path cannot be empty. Provide at least one path{suffix}.')
    if allow_global:
        if '**' in paths and len(paths) > 1:
            raise ValueError(f'"**" cannot be mixed with other paths. Pass "**" alone for {scope_hint}.')
    elif any(p == '**' for p in paths):
        raise ValueError('"**" is not supported here. Specify concrete paths.')
    return [resolve_search_path(p) for p in paths]


@overload
def to_repo_filter(value: str) -> str: ...
@overload
def to_repo_filter(value: Sequence[str]) -> Sequence[str]: ...
def to_repo_filter(value: str | Sequence[str]) -> str | Sequence[str]:
    """Translate the user-facing ``"**"`` sentinel to the repository's empty-value form.

    Pure translator. Validation belongs to ``resolve_search_path`` /
    ``resolve_search_paths``; this function trusts that its input has
    already been validated and just performs the boundary swap so the
    repository layer never sees ``"**"``.

    - str input: ``"**"`` → ``""``; otherwise unchanged.
    - Sequence[str] input: contains ``"**"`` → ``[]``; otherwise ``list(value)``.

    Compose at the boundary:

        filter_path = to_repo_filter(resolve_search_path(path))
        source_prefixes = to_repo_filter(
            resolve_search_paths(paths, scope_hint='global scope'),
        )

    Use when feeding a repository method (``get_content_stats``,
    ``list_indexed_files``) or constructing a ``SearchQuery``. For tools
    that propagate ``"**"`` to a service layer (e.g.
    ``IndexingService.clear_documents``) or that disallow ``"**"`` entirely
    (e.g. indexing), skip this translator and use the resolver output
    directly.
    """
    if isinstance(value, str):
        return '' if value == '**' else value
    return [] if '**' in value else list(value)


GLOB_CHARS: Set[str] = {
    '*',
    '?',
    '[',
    ']',
}
