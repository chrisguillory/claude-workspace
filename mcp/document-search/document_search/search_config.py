"""Search-recall constants and snippet formatting shared by the CLI and MCP surfaces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_search.schemas.vectors import SearchHit

__all__ = [
    'HYBRID_PREFETCH_FLOOR',
    'MAX_SEARCH_RESULTS',
    'PATH_FILTER_OVERFETCH_FACTOR',
    'RERANK_CANDIDATE_CAP',
    'RERANK_OVERFETCH_FACTOR',
    'clamp_search_limit',
    'format_snippet',
    'neighbor_chunk_targets',
    'rerank_candidate_count',
    'resolve_context_window',
    'snippet_hit',
]

RERANK_OVERFETCH_FACTOR = 3
"""Candidates fetched per requested result, to give the reranker depth."""

RERANK_CANDIDATE_CAP = 200
"""Hard ceiling on the rerank candidate pool (bounds the Qdrant fetch + cross-encoder cost)."""

PATH_FILTER_OVERFETCH_FACTOR = 3
"""Extra Qdrant fetch multiplier when post-filtering by path prefix (Qdrant can't prefix-match)."""

HYBRID_PREFETCH_FLOOR = 50
"""Floor for each hybrid prefetch branch (dense, sparse) before RRF fusion. Prefetch scales
with the requested candidate count but never drops below this, so small-N queries keep
fusion depth while large-N queries can still reach the candidate cap."""

MAX_SEARCH_RESULTS = 200
"""User-facing result cap — a product choice, bounded by ``RERANK_CANDIDATE_CAP`` (the
reranker can't return more results than its candidate pool, so this must stay ``<=`` it).
Set to that ceiling so a request can reach the system's true maximum."""


def clamp_search_limit(limit: int) -> int:
    """Clamp a requested result count into the supported range ``[1, MAX_SEARCH_RESULTS]``."""
    return min(max(limit, 1), MAX_SEARCH_RESULTS)


def rerank_candidate_count(effective_limit: int) -> int:
    """Candidate-pool size to fetch before reranking — over-fetched, then capped."""
    return min(effective_limit * RERANK_OVERFETCH_FACTOR, RERANK_CANDIDATE_CAP)


def resolve_context_window(*, context: int, before: int | None, after: int | None) -> tuple[int, int]:
    """Resolve the ``grep``-style ``-C``/``-B``/``-A`` flags into ``(before, after)`` counts.

    ``context`` sets both directions; an explicit ``before``/``after`` overrides its own
    direction (so ``-C 2 -A 0`` means two before, none after). All-zero is the no-context
    default. Mirrors ``grep``: ``-C N`` ≡ ``-B N -A N``. Both surfaces resolve through here
    so the CLI and MCP agree on precedence.
    """
    return (
        context if before is None else before,
        context if after is None else after,
    )


def neighbor_chunk_targets(
    hit_positions: Iterable[tuple[str, int]],
    *,
    before: int,
    after: int,
) -> Mapping[str, frozenset[int]]:
    """Compute which adjacent chunks to fetch as ``grep``-style context for the hits.

    Maps each ``source_path`` to the set of neighbor chunk indices to fetch. Boundary
    and dedup semantics:

    - Indices below ``0`` are dropped (no chunk precedes the first) — fail-soft at the
      lower document boundary. The upper boundary needs no clamp here: an index past the
      last chunk simply matches nothing in the store.
    - A neighbor that is itself a hit is excluded — the match is already returned as a
      hit, so re-emitting it as context would duplicate it.
    - Overlapping windows collapse: ``frozenset`` dedups, so a chunk in two hits' windows
      is fetched once.

    ``before == after == 0`` yields an empty mapping (the no-context default).
    """
    hit_positions = set(hit_positions)
    targets: defaultdict[str, set[int]] = defaultdict(set)
    for source_path, chunk_index in hit_positions:
        for offset in range(-before, after + 1):
            neighbor_index = chunk_index + offset
            if neighbor_index < 0:
                continue
            if (source_path, neighbor_index) in hit_positions:
                continue
            targets[source_path].add(neighbor_index)
    return {source_path: frozenset(indices) for source_path, indices in targets.items()}


def format_snippet(text: str, *, max_chars: int | None) -> str:
    """Shape a hit's text for display or return.

    ``None`` returns the text unchanged (full). A positive ``max_chars`` truncates and
    appends an ellipsis only when truncation actually occurred, so a truncated result is
    ``max_chars + 1`` characters. Newline handling is the caller's concern. Callers pass
    ``None`` or a value ``>= 1`` (enforced at the CLI/MCP boundary), never ``0``.
    """
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + '…'


def snippet_hit(hit: SearchHit, max_chars: int | None) -> SearchHit:
    """Truncate a hit's text and its neighboring-context texts to ``max_chars``.

    Returns a copy with ``text`` shaped by :func:`format_snippet` and the same applied to
    each ``before``/``after`` neighbor, so context chunks render at the same width as the
    hit. ``None`` leaves all text full.
    """
    return hit.__replace__(
        text=format_snippet(hit.text, max_chars=max_chars),
        before=[snippet_hit(n, max_chars) for n in hit.before],
        after=[snippet_hit(n, max_chars) for n in hit.after],
    )
