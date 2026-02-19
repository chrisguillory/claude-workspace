"""BM25 sparse embedding service using bm25-rs Rust extension.

Generates sparse vectors for keyword matching in hybrid search.
Uses Rust + rayon for parallel BM25 computation — releases the GIL
and distributes work across CPU cores without subprocess overhead.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import bm25_rs

__all__ = [
    'SparseEmbeddingService',
]

logger = logging.getLogger(__name__)

# Type alias for sparse vector result
type SparseVector = tuple[Sequence[int], Sequence[float]]


class SparseEmbeddingService:
    """BM25 sparse embedding service for keyword matching.

    Uses bm25-rs Rust extension with rayon parallelism. Single BM25Model
    instance shared across all async workers — thread safety comes from
    immutable model (&self) plus rayon thread-local mutable state.
    """

    def __init__(self) -> None:
        self._model = bm25_rs.BM25Model()
        self._thread_count = bm25_rs.BM25Model.thread_count()
        logger.debug(f'[SPARSE] rayon thread pool: {self._thread_count} threads')

    @property
    def thread_count(self) -> int:
        """Number of rayon worker threads available for parallel BM25 computation."""
        return self._thread_count

    @classmethod
    async def create(cls, **kwargs: object) -> SparseEmbeddingService:
        """Factory method. Accepts and ignores kwargs for API compatibility."""
        return cls()

    async def embed(self, text: str) -> SparseVector:
        """Generate sparse vector for a single text."""
        results, _wall, _cpu = await self.embed_batch(texts=[text])
        return results[0]

    async def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[Sequence[SparseVector], float, float]:
        """Generate sparse vectors for multiple texts.

        Runs BM25 in a thread via asyncio.to_thread to keep the event loop
        responsive. Rust releases the GIL internally, so the event loop
        thread can process I/O (httpx, Redis) while rayon computes.

        Timing is measured inside Rust (not Python perf_counter) so it
        captures only BM25 compute with zero noise from async scheduling.

        Args:
            texts: Texts to embed.

        Returns:
            (sparse_vectors, wall_secs, cpu_secs) — wall is latency,
            cpu is total compute across all rayon threads.
        """
        if not texts:
            return [], 0.0, 0.0
        # asyncio.to_thread prevents event loop blocking during PyO3 argument
        # conversion (list → Vec<String>). The Rust computation itself releases
        # the GIL via py.allow_threads, so only the conversion phase blocks.
        results, wall_secs, cpu_secs = await asyncio.to_thread(self._model.embed_batch, list(texts))
        parallel = cpu_secs / wall_secs if wall_secs > 0 else 0.0
        logger.debug(
            f'[SPARSE] {len(texts)} texts in {wall_secs:.3f}s wall, {cpu_secs:.3f}s cpu ({parallel:.1f}x parallel)'
        )
        return results, wall_secs, cpu_secs

    def shutdown(self) -> None:
        """No-op. Rayon thread pool is managed internally by Rust."""
