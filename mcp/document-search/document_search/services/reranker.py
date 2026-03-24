"""Cross-encoder reranking service using fastembed ONNX backend."""

from __future__ import annotations

import asyncio

from fastembed.rerank.cross_encoder import TextCrossEncoder

from document_search.schemas import SearchHit, SearchResult

__all__ = [
    'RerankerService',
]


class RerankerService:
    """Cross-encoder reranker using ms-marco model via ONNX Runtime."""

    MODEL_NAME = 'Xenova/ms-marco-MiniLM-L-6-v2'

    def __init__(self) -> None:
        """Initialize the reranker model."""
        self._encoder = TextCrossEncoder(model_name=self.MODEL_NAME)

    async def rerank(
        self,
        query: str,
        result: SearchResult,
        top_k: int | None = None,
    ) -> SearchResult:
        """Rerank search results using cross-encoder.

        Args:
            query: Original search query.
            result: Search results from hybrid search.
            top_k: Return only top k results after reranking (None = all).

        Returns:
            SearchResult with hits reordered by cross-encoder scores.
        """
        if not result.hits:
            return result

        documents = [hit.text for hit in result.hits]

        # fastembed rerank returns scores in document order (Iterable[float])
        scores = await asyncio.to_thread(
            lambda: list(self._encoder.rerank(query, documents)),
        )

        # Pair each hit with its reranker score, sort descending
        scored_hits = sorted(
            zip(scores, result.hits, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )

        reranked_hits: list[SearchHit] = [hit.model_copy(update={'score': score}) for score, hit in scored_hits]

        if top_k is not None:
            reranked_hits = reranked_hits[:top_k]

        return SearchResult(hits=reranked_hits, total=result.total)
