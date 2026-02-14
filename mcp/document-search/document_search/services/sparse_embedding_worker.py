"""BM25 sparse embedding worker for ProcessPoolExecutor.

Module-level functions that can be pickled and run in subprocesses.
All functions must be module-level (not methods) and use only primitives.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from fastembed import SparseTextEmbedding

__all__ = [
    'embed_batch',
]

MODEL_NAME = 'Qdrant/bm25'


def embed_batch(
    texts: Sequence[str],
) -> tuple[Sequence[tuple[Sequence[int], Sequence[float]]], float]:
    """Embed texts in subprocess - returns picklable types only.

    Args:
        texts: List of texts to embed.

    Returns:
        Tuple of (sparse_vectors, cpu_seconds). CPU time measured via
        time.process_time() in the subprocess for accurate profiling.
    """
    cpu_start = time.process_time()
    model = _get_model()
    results = list(model.embed(texts))
    cpu_seconds = time.process_time() - cpu_start
    return [(r.indices.tolist(), r.values.tolist()) for r in results], cpu_seconds


# Private module state and helpers below __all__ and public functions

# Global model instance per process (lazy init on first use)
_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    """Get or initialize model in subprocess."""
    global _model
    if _model is None:
        _model = SparseTextEmbedding(MODEL_NAME)
    return _model
