"""Embedding service - typed interface over Gemini client.

Translates between typed Pydantic models and the Gemini API.
"""

from __future__ import annotations

from document_search.clients.gemini import GeminiClient
from document_search.schemas.embeddings import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
)


class EmbeddingService:
    """Typed embedding service.

    Provides type-safe interface for embedding operations.
    Converts Pydantic models to/from Gemini API calls.
    """

    def __init__(self, client: GeminiClient) -> None:
        """Initialize service.

        Args:
            client: Gemini client.
        """
        self._client = client

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embed single text.

        Args:
            request: Typed embed request.

        Returns:
            Typed embed response.
        """
        vectors = await self._client.embed(
            texts=[request.text],
            task_type=request.task_type,
        )
        values = vectors[0]
        return EmbedResponse(values=tuple(values), dimensions=len(values))

    async def embed_batch(self, request: EmbedBatchRequest) -> EmbedBatchResponse:
        """Embed batch of texts.

        Args:
            request: Typed batch request (max 100 texts).

        Returns:
            Typed batch response.
        """
        vectors = await self._client.embed(
            texts=request.texts,
            task_type=request.task_type,
        )
        embeddings = [EmbedResponse(values=tuple(v), dimensions=len(v)) for v in vectors]
        return EmbedBatchResponse(embeddings=embeddings)
