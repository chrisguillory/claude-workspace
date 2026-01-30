"""Low-level Gemini API client.

Thin wrapper around google-genai. Handles API calls only - no business logic.
Type translation happens in the service layer.

Uses native async API (client.aio) for true concurrent requests.
Rate limiting via pyrate_limiter to respect API quotas.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import httpx
import pyrate_limiter
import tenacity
from google import genai
from google.genai.types import EmbedContentConfig, HttpOptions
from local_lib import ConcurrencyTracker

from document_search.clients import _retry
from document_search.schemas.embeddings import TaskIntent

__all__ = [
    'GeminiClient',
]

# Gemini-specific task types
type GeminiTaskType = Literal[
    'RETRIEVAL_DOCUMENT',
    'RETRIEVAL_QUERY',
    'SEMANTIC_SIMILARITY',
    'CLASSIFICATION',
    'CLUSTERING',
    'QUESTION_ANSWERING',
    'FACT_VERIFICATION',
]


class GeminiClient:
    """Low-level Gemini API client with rate limiting.

    Uses native async API (client.aio) for true concurrent HTTP requests.
    Rate limited via pyrate_limiter, concurrency controlled via semaphore.
    """

    # Rate limiting - Tier 1 limits:
    # - RPM: 3000 requests per minute
    # - TPM: 1,000,000 tokens per minute (this is usually the bottleneck)
    DEFAULT_REQUESTS_PER_MINUTE = 3000
    DEFAULT_TOKENS_PER_MINUTE = 1_000_000

    # Concurrency control - semaphore limits concurrent API calls
    DEFAULT_MAX_CONCURRENT = 200  # Tuned for throughput - Gemini handles well

    # HTTP client configuration - tuned to avoid PoolTimeout
    DEFAULT_TIMEOUT_MS = 5000  # request timeout in milliseconds (google-genai default)
    DEFAULT_MAX_CONNECTIONS = 200  # match semaphore to avoid connection queuing
    DEFAULT_MAX_KEEPALIVE = 200  # keep all connections alive for reuse
    DEFAULT_KEEPALIVE_EXPIRY = 30  # seconds before idle close

    def __init__(
        self,
        model: str,
        output_dimensionality: int,
        *,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        api_key: str | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        keepalive_expiry: float = DEFAULT_KEEPALIVE_EXPIRY,
    ) -> None:
        """Initialize client.

        Args:
            model: Embedding model name (e.g., 'gemini-embedding-001').
            output_dimensionality: Output vector dimensions (e.g., 768).
            requests_per_minute: Rate limit (default 3000 for Tier 1).
            api_key: Gemini API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests (semaphore limit).
            timeout_ms: Request timeout in milliseconds.
            max_connections: Max simultaneous HTTP connections.
            max_keepalive: Max connections kept alive for reuse.
            keepalive_expiry: Seconds before idle connections close.
        """
        self._model = model
        self._output_dimensionality = output_dimensionality
        self._api_key = api_key or _load_api_key()

        # RPM limiter: 100 texts per (time to do one batch at rpm)
        # 3000/min = 30 batches/min = 1 batch per 2 seconds
        batch_size = 100
        batches_per_minute = requests_per_minute // batch_size  # 3000/100 = 30
        seconds_per_batch = 60 // batches_per_minute  # 60/30 = 2 seconds
        self._rpm_limiter = pyrate_limiter.Limiter(
            pyrate_limiter.Rate(batch_size, seconds_per_batch * pyrate_limiter.Duration.SECOND),
        )

        # TPM limiter: 1M tokens/min, 12-second window.
        # 200K budget fits exactly 4 batches of 50K tokens = 100% utilization. 429s trigger retry.
        tokens_per_window = self.DEFAULT_TOKENS_PER_MINUTE // 5  # 200K per 12s
        self._tpm_limiter = pyrate_limiter.Limiter(
            pyrate_limiter.Rate(tokens_per_window, 12 * pyrate_limiter.Duration.SECOND),
        )

        # HTTP client configuration
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

    # Gemini task type mapping from generic intent
    INTENT_TO_GEMINI_TASK: Mapping[TaskIntent, GeminiTaskType] = {
        'document': 'RETRIEVAL_DOCUMENT',
        'query': 'RETRIEVAL_QUERY',
    }

    @_retry.gemini_breaker
    @tenacity.retry(
        retry=tenacity.retry_if_exception(_retry.is_retryable_gemini_error),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=0.5, max=5),
        before_sleep=_retry.log_gemini_retry,
    )
    async def embed(
        self,
        texts: Sequence[str],
        *,
        intent: TaskIntent,
    ) -> Sequence[Sequence[float]]:
        """Embed texts using Gemini API.

        Uses native async API for true concurrent requests.
        Rate limiting handled internally via semaphore.
        Retries on transient network errors (ReadError, WriteError, timeouts).

        Args:
            texts: Texts to embed (max 100 per API call).
            intent: 'document' for indexing, 'query' for search.

        Returns:
            List of embedding vectors.

        Raises:
            google.genai.errors.ClientError: On API errors.
        """
        # Estimate tokens: ~0.513 tokens/char matches Google's dashboard exactly.
        # Calibrated by comparing our TPM to Google AI Studio's display.
        estimated_tokens = int(sum(len(t) for t in texts) / 1.95)

        # Acquire from both limiters (RPM by text count, TPM by token estimate)
        await self._rpm_limiter.try_acquire_async('rpm', weight=len(texts))
        await self._tpm_limiter.try_acquire_async('tpm', weight=estimated_tokens)

        # Translate intent to Gemini's task_type
        gemini_task_type = self.INTENT_TO_GEMINI_TASK[intent]

        async with self._semaphore, self._tracker.track():
            result = await self._client.aio.models.embed_content(
                model=self._model,
                contents=list(texts),
                config=EmbedContentConfig(
                    task_type=gemini_task_type, output_dimensionality=self._output_dimensionality
                ),
            )
            return [list(e.values) for e in result.embeddings]

    async def close(self) -> None:
        """No-op: google-genai Client manages its own HTTP lifecycle."""


def _load_api_key() -> str:
    """Load API key from standard location."""
    key_path = Path.home() / '.claude-workspace' / 'secrets' / 'document_search_api_key'
    if not key_path.exists():
        raise FileNotFoundError(f'API key not found at {key_path}')
    return key_path.read_text().strip()
