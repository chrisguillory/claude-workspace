"""BM25 sparse embedding service using fastembed.

Generates sparse vectors for keyword matching in hybrid search.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import SparseTextEmbedding


class SparseEmbeddingService:
    """BM25 sparse embedding service for keyword matching.

    Uses fastembed's Qdrant/bm25 model to generate sparse vectors
    compatible with Qdrant's sparse vector index.
    """

    MODEL_NAME = 'Qdrant/bm25'

    def __init__(self) -> None:
        """Initialize the BM25 sparse embedding model.

        Note: First call triggers model download (~50MB).
        """
        self._model = SparseTextEmbedding(self.MODEL_NAME)

    def embed(self, text: str) -> tuple[Sequence[int], Sequence[float]]:
        """Generate sparse vector for a single text.

        Args:
            text: Text to embed.

        Returns:
            Tuple of (indices, values) for sparse vector.
        """
        results = list(self._model.embed([text]))
        result = results[0]
        return result.indices.tolist(), result.values.tolist()

    def embed_batch(self, texts: Sequence[str]) -> Sequence[tuple[Sequence[int], Sequence[float]]]:
        """Generate sparse vectors for multiple texts.

        Args:
            texts: Texts to embed.

        Returns:
            Sequence of (indices, values) tuples for sparse vectors.
        """
        results = list(self._model.embed(list(texts)))
        return [(r.indices.tolist(), r.values.tolist()) for r in results]
