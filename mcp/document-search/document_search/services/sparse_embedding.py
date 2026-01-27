"""BM25 sparse embedding service using fastembed.

Generates sparse vectors for keyword matching in hybrid search.
Uses ProcessPoolExecutor to bypass GIL for CPU-bound BM25 work.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from document_search.services import sparse_embedding_worker

__all__ = [
    'SparseEmbeddingService',
]

# Type alias for sparse vector result
type SparseVector = tuple[Sequence[int], Sequence[float]]


class SparseEmbeddingService:
    """BM25 sparse embedding service for keyword matching.

    Uses ProcessPoolExecutor for parallel BM25 processing across CPU cores.
    """

    @classmethod
    async def create(cls, *, workers: int | None = None) -> SparseEmbeddingService:
        """Create service with ProcessPoolExecutor.

        Args:
            workers: Max worker processes. Defaults to cpu_count.
        """
        if workers is None:
            workers = os.cpu_count() or 4
        return cls(_process_pool=ProcessPoolExecutor(max_workers=workers))

    def __init__(self, *, _process_pool: ProcessPoolExecutor) -> None:
        """Internal - use create() factory."""
        self._pool = _process_pool

    async def embed(self, text: str) -> SparseVector:
        """Generate sparse vector for a single text.

        Args:
            text: Text to embed.

        Returns:
            Tuple of (indices, values) for sparse vector.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        """Generate sparse vectors for multiple texts in subprocess.

        Args:
            texts: Texts to embed.

        Returns:
            Sequence of (indices, values) tuples for sparse vectors.
        """
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            self._pool,
            sparse_embedding_worker.embed_batch,
            list(texts),
        )
        return results

    def shutdown(self) -> None:
        """Shutdown ProcessPoolExecutor and release resources."""
        self._pool.shutdown(wait=True)
