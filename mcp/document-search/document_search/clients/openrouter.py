"""OpenRouter embedding client.

Thin wrapper around the OpenRouter embeddings API. Handles API calls only - no business logic.
Type translation happens in the service layer.

Uses native async httpx for concurrent requests.
Concurrency controlled via semaphore.

API Reference: https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
import pydantic
import tenacity
from cc_lib import ConcurrencyTracker
from cc_lib.types import JsonObject

from document_search.clients import _retry
from document_search.clients.openrouter_errors import OpenRouterAPIError, OpenRouterUnexpectedResponse
from document_search.schemas.embeddings import TaskIntent

__all__ = [
    'OpenRouterClient',
]

logger = logging.getLogger(__name__)


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
        encoding_format: Literal['float', 'base64'] = 'base64',
    ) -> None:
        """Initialize client.

        Args:
            model: Model identifier (e.g., 'qwen/qwen3-embedding-8b').
            dimensions: Output vector dimensions. If None, uses model's native dimensions.
                Note: Not all models support dimension reduction.
            requests_per_minute: Not implemented yet. Accepted but ignored.
            api_key: OpenRouter API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests (semaphore limit).
            timeout_ms: Request timeout in milliseconds.
            max_connections: Max simultaneous HTTP connections.
            max_keepalive: Max connections kept alive for reuse.
            keepalive_expiry: Seconds before idle connections close.
            encoding_format: Response format. 'base64' (default) is ~47%% smaller on the wire.
        """
        if requests_per_minute is not None:
            logger.info(f'requests_per_minute={requests_per_minute} accepted but not yet enforced')
        self._model = model
        self._total_tokens = 0
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

            # HTTP 4xx/5xx — retry predicate handles via httpx.HTTPStatusError
            response.raise_for_status()

            # Content-type guard — catch non-JSON (Cloudflare pages, etc.)
            content_type = response.headers.get('content-type', '')
            if content_type and not content_type.startswith('application/json'):
                raise OpenRouterUnexpectedResponse(
                    message=f'Expected application/json, got {content_type}',
                    body_preview=response.text[:500],
                    body_keys=[],
                    status_code=response.status_code,
                    content_type=content_type,
                    model=self._model,
                    batch_size=len(texts),
                )

            # Parse JSON — JSONDecodeError is retryable (truncated body = transient)
            data = response.json()

            # Three-way discrimination via Pydantic
            result = _parse_embedding_response(
                data,
                status_code=response.status_code,
                content_type=content_type,
                model=self._model,
                batch_size=len(texts),
            )

            self._total_tokens += result.usage.total_tokens

            sorted_data = sorted(result.data, key=lambda x: x.index)
            return [_decode_embedding(e.embedding) for e in sorted_data]

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens consumed across all embed() calls."""
        return self._total_tokens

    async def list_models(
        self,
    ) -> Sequence[JsonObject]:
        """List all available embedding models.

        Returns model metadata including id, name, pricing, context_length, etc.

        API Reference: https://openrouter.ai/docs/api/api-reference/embeddings/list-embeddings-models

        Returns:
            List of model info dicts with keys: id, name, pricing, context_length,
            architecture, etc.

        Raises:
            httpx.HTTPStatusError: On non-retryable API errors.
            OpenRouterAPIError: On error response body.
        """
        response = await self._client.get('/embeddings/models')
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        if 'error' in data:
            try:
                error_resp = _ErrorResponse.model_validate(data)
            except pydantic.ValidationError as e:
                raise OpenRouterUnexpectedResponse(
                    message='list_models: error key with unexpected format',
                    body_preview=json.dumps(data, default=str)[:500],
                    body_keys=list(data.keys()),
                    status_code=response.status_code,
                    content_type=response.headers.get('content-type', ''),
                    model=self._model,
                    batch_size=0,
                ) from e
            raise OpenRouterAPIError(
                message=error_resp.error.message,
                code=error_resp.error.code,
                error_type=error_resp.error.type,
                status_code=response.status_code,
                model=self._model,
            )
        models: Sequence[JsonObject] = data['data']
        return models

    async def close(self) -> None:
        """Close HTTP client and release resources."""
        await self._client.aclose()

    async def __aenter__(self) -> OpenRouterClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Async context manager exit."""
        await self.close()


# ── OpenRouter API response models ───────────────────────────────────


class _Embedding(pydantic.BaseModel):
    embedding: Sequence[float] | str
    index: int


class _Usage(pydantic.BaseModel):
    prompt_tokens: int
    total_tokens: int


class _EmbeddingResponse(pydantic.BaseModel):
    data: Sequence[_Embedding]
    model: str
    usage: _Usage


class _ErrorDetail(pydantic.BaseModel):
    message: str
    code: int | None = None
    type: str | None = None
    metadata: Mapping[str, Any] | None = (
        None  # strict_typing_linter.py: loose-typing — provider-specific metadata with no stable schema
    )


class _ErrorResponse(pydantic.BaseModel):
    error: _ErrorDetail


type _ApiResponse = _EmbeddingResponse | _ErrorResponse

_api_response_adapter: pydantic.TypeAdapter[_ApiResponse] = pydantic.TypeAdapter(_ApiResponse)


# ── Private helpers ──────────────────────────────────────────────────


def _load_api_key() -> str:
    """Load API key from standard location."""
    key_path = Path.home() / '.claude-workspace' / 'secrets' / 'openrouter_api_key'
    if not key_path.exists():
        raise FileNotFoundError(f'OpenRouter API key not found at {key_path}')
    return key_path.read_text().strip()


def _parse_embedding_response(
    body: Mapping[str, Any],
    *,
    status_code: int,
    content_type: str,
    model: str,
    batch_size: int,
) -> _EmbeddingResponse:
    """Validate and discriminate API response via Pydantic union."""
    try:
        result = _api_response_adapter.validate_python(body)
    except pydantic.ValidationError as e:
        raise OpenRouterUnexpectedResponse(
            message=f'Response validation failed: {e.error_count()} errors',
            body_preview=json.dumps(body, default=str)[:500],
            body_keys=list(body.keys()),
            status_code=status_code,
            content_type=content_type,
            model=model,
            batch_size=batch_size,
        ) from e

    if isinstance(result, _ErrorResponse):
        raise OpenRouterAPIError(
            message=result.error.message,
            code=result.error.code,
            error_type=result.error.type,
            status_code=status_code,
            model=model,
        )

    return result


def _decode_embedding(embedding: Sequence[float] | str) -> Sequence[float]:
    """Decode from float array or base64 string."""
    if isinstance(embedding, str):
        values: Sequence[float] = np.frombuffer(base64.b64decode(embedding), dtype=np.float32).tolist()
        return values
    return list(embedding)
