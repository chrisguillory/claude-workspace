"""Low-level Gemini API client.

Thin wrapper around google-genai. Handles API calls only - no business logic.
Type translation happens in the service layer.

Rate limiting is handled internally via semaphore - callers can fire all
requests without worrying about concurrency limits.
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

    Concurrency is controlled internally via semaphore. Callers can fire
    unlimited concurrent requests - the client handles queuing.
    """

    DEFAULT_MODEL = 'text-embedding-004'
    DEFAULT_MAX_CONCURRENT = 1_000  # High limit; let API rate limits be the constraint

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    ) -> None:
        """Initialize client.

        Args:
            api_key: Gemini API key. If None, loads from standard location.
            max_concurrent: Max concurrent API requests (default 5).
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
    ) -> list[list[float]]:
        """Embed texts using Gemini API.

        Rate limiting handled internally - callers can fire unlimited
        concurrent requests without worrying about API limits.

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
            # Run blocking SDK call in thread pool
            return await asyncio.to_thread(
                self._embed_sync,
                texts,
                task_type=task_type,
                model=model,
            )

    def _embed_sync(
        self,
        texts: Sequence[str],
        *,
        task_type: str,
        model: str,
    ) -> list[list[float]]:
        """Synchronous embed implementation."""
        result = self._client.models.embed_content(
            model=model,
            contents=list(texts),
            config={'task_type': task_type},
        )
        return [list(e.values) for e in result.embeddings]
