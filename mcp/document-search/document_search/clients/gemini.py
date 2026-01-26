"""Low-level Gemini API client.

Thin wrapper around google-genai. Handles API calls only - no business logic.
Type translation happens in the service layer.

Uses native async API (client.aio) for true concurrent requests.
Rate limiting is handled internally via semaphore.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import httpx
import tenacity
from google import genai
from google.genai.types import HttpOptions
from local_lib import ConcurrencyTracker

from document_search.clients import _gemini_retry

__all__ = [
    'GeminiClient',
]


class GeminiClient:
    """Low-level Gemini API client with internal rate limiting.

    Uses native async API (client.aio) for true concurrent HTTP requests.
    Concurrency is controlled via semaphore to respect API rate limits.
    """

    DEFAULT_MODEL = 'text-embedding-004'

    # Concurrency control - semaphore limits concurrent API calls
    DEFAULT_MAX_CONCURRENT = 200  # Tuned for throughput - observed max ~210

    # HTTP client configuration (google-genai defaults, explicit for tuning)
    DEFAULT_TIMEOUT_MS = 5000  # request timeout in milliseconds (google-genai default)
    DEFAULT_MAX_CONNECTIONS = 100  # max simultaneous connections (google-genai default)
    DEFAULT_MAX_KEEPALIVE = 20  # connections kept alive for reuse (google-genai default)
    DEFAULT_KEEPALIVE_EXPIRY = 5.0  # seconds before idle close (google-genai default)

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
    ) -> None:
        """Initialize client.

        Args:
            api_key: Gemini API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests (semaphore limit).
            timeout_ms: Request timeout in milliseconds.
            max_connections: Max simultaneous HTTP connections.
            max_keepalive: Max connections kept alive for reuse.
            keepalive_expiry: Seconds before idle connections close.
        """
        self._api_key = api_key or _load_api_key()

        # Explicit HTTP client configuration for observability and tuning
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )
        http_options = HttpOptions(
            timeout=timeout_ms,
            async_client_args={'limits': limits},
        )
        self._client = genai.Client(api_key=self._api_key, http_options=http_options)

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tracker = ConcurrencyTracker('GEMINI')

    @tenacity.retry(
        retry=tenacity.retry_if_exception(_gemini_retry.is_retryable_gemini_error),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=0.5, max=5),
        before_sleep=_gemini_retry.log_gemini_retry,
    )
    async def embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str = 'RETRIEVAL_DOCUMENT',
        model: str = DEFAULT_MODEL,
    ) -> Sequence[Sequence[float]]:
        """Embed texts using Gemini API.

        Uses native async API for true concurrent requests.
        Rate limiting handled internally via semaphore.
        Retries on transient network errors (ReadError, WriteError, timeouts).

        Args:
            texts: Texts to embed (max 100 per API call).
            task_type: Embedding task type.
            model: Model name.

        Returns:
            List of embedding vectors.

        Raises:
            google.genai.errors.ClientError: On API errors.
        """
        async with self._semaphore, self._tracker.track():
            result = await self._client.aio.models.embed_content(
                model=model,
                contents=list(texts),
                config={'task_type': task_type},
            )
            return [list(e.values) for e in result.embeddings]


def _load_api_key() -> str:
    """Load API key from standard location."""
    key_path = Path.home() / '.claude-workspace' / 'secrets' / 'document_search_api_key'
    if not key_path.exists():
        raise FileNotFoundError(f'API key not found at {key_path}')
    return key_path.read_text().strip()
