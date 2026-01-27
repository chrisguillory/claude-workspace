"""Cross-encoder reranking service."""

from __future__ import annotations

import asyncio

from rerankers import Reranker

from document_search.schemas import SearchHit, SearchResult

__all__ = [
    'RerankerService',
]


class RerankerService:
    """Cross-encoder reranker using ms-marco model."""

    MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    def __init__(self) -> None:
        """Initialize the reranker model."""
        self._reranker = Reranker(self.MODEL_NAME, verbose=0)

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

        # Prepare documents for reranking
        documents = [hit.text for hit in result.hits]

        # Run blocking ML inference in thread pool to avoid blocking event loop
        ranked = await asyncio.to_thread(self._reranker.rank, query=query, docs=documents)

        # Reorder hits by reranker scores
        # ranked.results contains Result objects with doc_id (index) and score
        reranked_hits: list[SearchHit] = []
        for ranked_result in ranked.results:
            original_hit = result.hits[ranked_result.doc_id]
            # Update score while preserving all other fields
            reranked_hits.append(original_hit.model_copy(update={'score': ranked_result.score}))

        # Apply top_k limit if specified
        if top_k is not None:
            reranked_hits = reranked_hits[:top_k]

        return SearchResult(hits=reranked_hits, total=result.total)
