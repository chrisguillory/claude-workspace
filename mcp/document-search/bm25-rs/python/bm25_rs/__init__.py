"""BM25 sparse embedding using Rust + rayon parallelism."""

from __future__ import annotations

from bm25_rs.bm25_rs import BM25Model

__all__ = ['BM25Model']
