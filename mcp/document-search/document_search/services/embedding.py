"""Embedding service - typed interface over embedding clients.

Translates between typed Pydantic models and the embedding API.
Uses BatchLoader pattern for automatic request coalescing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from local_lib.batch_loader import GenericBatchLoader

from document_search.clients.protocols import EmbeddingClient
from document_search.schemas.embeddings import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
)

__all__ = [
    'EmbeddingService',
]

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Typed embedding service with automatic batching.

    Provides type-safe interface for embedding operations.
    Uses internal BatchLoader to coalesce concurrent requests.
    """

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        batch_size: int,
        coalesce_delay: float = 0.01,
    ) -> None:
        """Initialize service.

        Args:
            client: Embedding client.
            batch_size: Max texts per batch.
            coalesce_delay: Seconds to wait for request coalescing (default 10ms).
        """
        self._client = client
        self._batch_size = batch_size
        self._loader = _EmbedLoader(self, batch_size=batch_size, coalesce_delay=coalesce_delay)

    async def embed_text(self, text: str) -> EmbedResponse:
        """Embed single text with automatic batching.

        Requests are coalesced with concurrent calls for efficiency.

        Args:
            text: Text to embed.

        Returns:
            Typed embed response.
        """
        return await self._loader.load(text)

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embed single text.

        Args:
            request: Typed embed request.

        Returns:
            Typed embed response.
        """
        vectors = await self._client.embed(
            texts=[request.text],
            intent=request.intent,
        )
        values = vectors[0]
        return EmbedResponse(values=values, dimensions=len(values))

    async def embed_batch(self, request: EmbedBatchRequest) -> EmbedBatchResponse:
        """Embed batch of texts.

        Args:
            request: Typed batch request.

        Returns:
            Typed batch response.
        """
        vectors = await self._client.embed(
            texts=request.texts,
            intent=request.intent,
        )
        embeddings = [EmbedResponse(values=v, dimensions=len(v)) for v in vectors]
        return EmbedBatchResponse(embeddings=embeddings)


class _EmbedLoader(GenericBatchLoader[str, EmbedResponse]):
    """Internal batch loader for embedding requests.

    Coalesces individual text embedding requests into batches.
    Uses text string as request key for deduplication.
    """

    def __init__(self, service: EmbeddingService, *, batch_size: int, coalesce_delay: float = 0.01) -> None:
        self._service = service
        super().__init__(
            bulk_load=self._bulk_embed,
            batch_size=batch_size,
            coalesce_delay=coalesce_delay,
        )

    async def _bulk_embed(self, texts: Sequence[str]) -> Sequence[EmbedResponse]:
        """Embed a batch of texts."""
        total_chars = sum(len(t) for t in texts)
        logger.debug(f'[BATCH] Embedding {len(texts)} texts ({total_chars:,} chars)')
        request = EmbedBatchRequest(texts=texts, intent='document')
        response = await self._service.embed_batch(request)
        return response.embeddings
