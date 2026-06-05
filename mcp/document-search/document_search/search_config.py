"""Search-recall constants and snippet formatting shared by the CLI and MCP surfaces."""

from __future__ import annotations

__all__ = [
    'HYBRID_PREFETCH_FLOOR',
    'MAX_SEARCH_RESULTS',
    'PATH_FILTER_OVERFETCH_FACTOR',
    'RERANK_CANDIDATE_CAP',
    'RERANK_OVERFETCH_FACTOR',
    'clamp_search_limit',
    'format_snippet',
    'rerank_candidate_count',
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
