"""Embedding batch loader - coalesces embedding requests into efficient batches.

Wraps GenericBatchLoader with Gemini API constraints (100 items max per batch).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from local_lib.batch_loader import GenericBatchLoader

from document_search.schemas.embeddings import (
    EmbedBatchRequest,
    EmbedResponse,
)
from document_search.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# Gemini API limit
GEMINI_BATCH_SIZE = 100


class EmbeddingBatchLoader(GenericBatchLoader[str, EmbedResponse]):
    """Batch loader for embeddings with Gemini API constraints.

    Coalesces individual text embedding requests into batches of up to 100.
    Uses text string as request key for deduplication.
    """

    def __init__(
        self,
        service: EmbeddingService,
        *,
        coalesce_delay: float = 0.01,
    ) -> None:
        """Initialize embedding batch loader.

        Args:
            service: Embedding service for API calls.
            coalesce_delay: Seconds to wait for more requests (default 10ms).
        """
        self._service = service
        super().__init__(
            bulk_load=self._bulk_embed,
            batch_size=GEMINI_BATCH_SIZE,
            coalesce_delay=coalesce_delay,
        )

    async def _bulk_embed(self, texts: Sequence[str]) -> Sequence[EmbedResponse]:
        """Embed a batch of texts."""
        total_chars = sum(len(t) for t in texts)
        logger.debug(f'[BATCH] Embedding {len(texts)} texts ({total_chars:,} chars)')
        request = EmbedBatchRequest(texts=texts)
        response = await self._service.embed_batch(request)
        return response.embeddings

    async def embed(self, text: str) -> EmbedResponse:
        """Embed a single text, batched with concurrent requests.

        Args:
            text: Text to embed.

        Returns:
            Embedding response.
        """
        return await self.load(text)
