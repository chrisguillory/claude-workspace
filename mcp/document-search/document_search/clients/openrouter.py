"""OpenRouter embedding client.

Thin wrapper around the OpenRouter embeddings API. Handles API calls only - no business logic.
Type translation happens in the service layer.

Uses native async httpx for concurrent requests.
Concurrency controlled via semaphore.

API Reference: https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import tenacity
from local_lib import ConcurrencyTracker

from document_search.clients import _retry
from document_search.schemas.embeddings import TaskIntent

__all__ = [
    'OpenRouterClient',
]


class OpenRouterClient:
    """Low-level OpenRouter embedding client.

    Uses native async httpx for concurrent HTTP requests.
    Concurrency controlled via semaphore.

    API Reference:
    - Embeddings: https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings
    - Models: https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models
    """

    BASE_URL = 'https://openrouter.ai/api/v1'

    # Qwen3-Embedding instruction prefix for query embeddings (asymmetric retrieval)
    # See: https://github.com/QwenLM/Qwen3-Embedding
    # - Queries: Prepend instruction prefix
    # - Documents: No prefix needed
    QUERY_INSTRUCTION = 'Instruct: Given a query, retrieve relevant passages that answer the query\nQuery:'

    # Concurrency control - semaphore limits concurrent API calls
    DEFAULT_MAX_CONCURRENT = 200  # Match Gemini

    # HTTP client configuration - match Gemini's tuned settings
    DEFAULT_TIMEOUT_MS = 5000  # Request timeout in milliseconds
    DEFAULT_MAX_CONNECTIONS = 200
    DEFAULT_MAX_KEEPALIVE = 200
    DEFAULT_KEEPALIVE_EXPIRY = 30  # Seconds before idle close

    def __init__(
        self,
        model: str,
        *,
        dimensions: int | None = None,
        requests_per_minute: int | None = None,
        api_key: str | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
        encoding_format: Literal['float', 'base64'] = 'float',
    ) -> None:
        """Initialize client.

        Args:
            model: Model identifier (e.g., 'qwen/qwen3-embedding-8b').
            dimensions: Output vector dimensions. If None, uses model's native dimensions.
                Note: Not all models support dimension reduction.
            requests_per_minute: Not supported at the moment.
            api_key: OpenRouter API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests (semaphore limit).
            timeout_ms: Request timeout in milliseconds.
            max_connections: Max simultaneous HTTP connections.
            max_keepalive: Max connections kept alive for reuse.
            keepalive_expiry: Seconds before idle connections close.
            encoding_format: Response format ('float' or 'base64').
        """
        if requests_per_minute is not None:
            raise ValueError('OpenRouterClient does not support rate limiting')
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key or _load_api_key()
        self._encoding_format = encoding_format

        # HTTP client with connection pooling
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
            },
            timeout=timeout_ms / 1000,  # Convert to seconds for httpx
            limits=limits,
        )

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tracker = ConcurrencyTracker('OPENROUTER')
        self.errors_429 = 0

    @_retry.openrouter_breaker
    @tenacity.retry(
        retry=tenacity.retry_if_exception(_retry.is_retryable_openrouter_error),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=0.5, max=5),
        before_sleep=_retry.log_openrouter_retry,
    )
    async def embed(
        self,
        texts: Sequence[str],
        *,
        intent: TaskIntent,
    ) -> Sequence[Sequence[float]]:
        """Embed texts using OpenRouter API.

        Uses native async httpx for concurrent requests.
        Concurrency controlled via semaphore.
        Retries on transient network errors and 429/502/503 responses.

        For Qwen3-Embedding models, applies instruction prefix for query embeddings
        to enable asymmetric retrieval (queries treated differently from documents).

        Args:
            texts: Texts to embed.
            intent: 'document' for indexing, 'query' for search.
                Query embeddings get an instruction prefix for asymmetric retrieval.

        Returns:
            List of embedding vectors.

        Raises:
            httpx.HTTPStatusError: On non-retryable API errors.
        """
        # Apply instruction prefix for query embeddings (asymmetric retrieval)
        if intent == 'query':
            texts = [f'{self.QUERY_INSTRUCTION}{t}' for t in texts]

        # Build request body per OpenAPI spec
        body: dict[str, object] = {
            'model': self._model,
            'input': list(texts),
            'encoding_format': self._encoding_format,
        }
        if self._dimensions is not None:
            body['dimensions'] = self._dimensions

        async with self._semaphore, self._tracker.track():
            response = await self._client.post('/embeddings', json=body)
            response.raise_for_status()
            data = response.json()

            # Sort by index to ensure order matches input
            embeddings = sorted(data['data'], key=lambda x: x['index'])
            return [e['embedding'] for e in embeddings]

    async def list_models(  # strict_typing_linter.py: loose-typing
        self,
    ) -> Sequence[Mapping[str, Any]]:
        """List all available embedding models.

        Returns model metadata including id, name, pricing, context_length, etc.

        API Reference: https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models

        Returns:
            List of model info dicts with keys: id, name, pricing, context_length,
            architecture, etc.

        Raises:
            httpx.HTTPStatusError: On API errors.
        """
        response = await self._client.get('/embeddings/models')
        response.raise_for_status()
        data: dict[str, list[Mapping[str, Any]]] = response.json()
        return data['data']

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenRouterClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Async context manager exit."""
        await self.close()


def _load_api_key() -> str:
    """Load API key from standard location."""
    key_path = Path.home() / '.claude-workspace' / 'secrets' / 'openrouter_api_key'
    if not key_path.exists():
        raise FileNotFoundError(f'OpenRouter API key not found at {key_path}')
    return key_path.read_text().strip()
