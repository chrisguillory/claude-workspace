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

from google import genai


def _load_api_key() -> str:
    """Load API key from standard location."""
    key_path = Path.home() / '.claude-workspace' / 'secrets' / 'document_search_api_key'
    if not key_path.exists():
        raise FileNotFoundError(f'API key not found at {key_path}')
    return key_path.read_text().strip()


class GeminiClient:
    """Low-level Gemini API client with internal rate limiting.

    Uses native async API (client.aio) for true concurrent HTTP requests.
    Concurrency is controlled via semaphore to respect API rate limits.
    """

    DEFAULT_MODEL = 'text-embedding-004'
    DEFAULT_MAX_CONCURRENT = 200  # Aggressive - may hit rate limits

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Initialize client.

        Args:
            api_key: Gemini API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests. Default 50 for Tier 1.
        """
        self._api_key = api_key or _load_api_key()
        self._client = genai.Client(api_key=self._api_key)
        self._semaphore = asyncio.Semaphore(max_concurrent)

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

        Args:
            texts: Texts to embed (max 100 per API call).
            task_type: Embedding task type.
            model: Model name.

        Returns:
            List of embedding vectors.

        Raises:
            google.genai.errors.ClientError: On API errors.
        """
        async with self._semaphore:
            result = await self._client.aio.models.embed_content(
                model=model,
                contents=list(texts),
                config={'task_type': task_type},
            )
            return [list(e.values) for e in result.embeddings]
