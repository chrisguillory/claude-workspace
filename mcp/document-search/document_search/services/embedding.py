"""Embedding service - typed interface over embedding clients.

Two-stage BatchLoader pipeline:
1. CacheLoader coalesces lookups into Redis MGET batches (100 batch, 1ms delay)
2. Cache misses forward to EmbedLoader for API batches (100-1000 batch, 10ms delay)

Cache values use numpy float32 binary (~3KB per 768-dim embedding vs ~17KB JSON).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Sequence

import numpy as np
from local_lib.background_tasks import BackgroundTaskGroup
from local_lib.batch_loader import GenericBatchLoader

from document_search.clients.protocols import EmbeddingClient
from document_search.clients.redis import RedisClient
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

# Cache configuration
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
CACHE_BATCH_SIZE = 100
CACHE_COALESCE_DELAY = 0.001  # 1ms - Redis is fast


class EmbeddingService:
    """Typed embedding service with Redis-backed caching and automatic batching.

    Two-stage BatchLoader pipeline:
    1. CacheLoader coalesces lookups into Redis MGET batches (100 batch, 1ms delay)
    2. Cache misses forward to EmbedLoader for API batches (100-1000 batch, 10ms delay)
    """

    def __init__(
        self,
        client: EmbeddingClient,
        *,
        batch_size: int,
        coalesce_delay: float = 0.01,
        redis: RedisClient,
        cache_tasks: BackgroundTaskGroup,
        model: str,
        dimensions: int,
    ) -> None:
        """Initialize service.

        Args:
            client: Embedding client.
            batch_size: Max texts per API batch.
            coalesce_delay: Seconds to wait for request coalescing (default 10ms).
            redis: Redis client for embedding cache.
            cache_tasks: Background task group for fire-and-forget cache writes.
            model: Embedding model name (for cache keys).
            dimensions: Embedding dimensions (for cache keys).
        """
        self._client = client
        self._batch_size = batch_size
        self._embed_loader = _EmbedLoader(self, batch_size=batch_size, coalesce_delay=coalesce_delay)
        self._cache_loader = CacheLoader(
            redis=redis,
            model=model,
            dimensions=dimensions,
            embed_loader=self._embed_loader,
            cache_tasks=cache_tasks,
        )

    @property
    def cache_hits(self) -> int:
        """Cumulative cache hits across all embed_text() calls."""
        return self._cache_loader.hits

    @property
    def cache_misses(self) -> int:
        """Cumulative cache misses across all embed_text() calls."""
        return self._cache_loader.misses

    async def embed_text(self, text: str) -> EmbedResponse:
        """Embed single text with automatic batching.

        Checks Redis first. Cache hits return immediately.
        Misses forward to embed loader for API batches.

        Args:
            text: Text to embed.

        Returns:
            Typed embed response.
        """
        return await self._cache_loader.load(text)

    async def embed_texts(self, texts: Sequence[str]) -> Sequence[EmbedResponse]:
        """Embed multiple texts as a single batch-aware call.

        Same cache/embed pipeline as embed_text(), but submits all items
        in one lock acquisition — no per-text Task creation on the event loop.

        Args:
            texts: Texts to embed.

        Returns:
            Typed embed responses in same order as texts.
        """
        return await self._cache_loader.load_many(texts)

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


class CacheLoader(GenericBatchLoader[str, EmbedResponse]):
    """Batch cache lookups with miss forwarding.

    Coalesces concurrent cache lookups into Redis MGET batches.
    Cache hits return immediately. Misses are forwarded to the
    embed loader for coalescing into API batches.

    Cache key format: embed:{model}:{dims}:document:{sha256[:16]}
    Intent is always 'document' since only embed_text() uses this path.
    """

    def __init__(
        self,
        redis: RedisClient,
        model: str,
        dimensions: int,
        embed_loader: _EmbedLoader,
        cache_tasks: BackgroundTaskGroup,
    ) -> None:
        self._redis = redis
        self._model = model
        self._dimensions = dimensions
        self._embed_loader = embed_loader
        self._cache_tasks = cache_tasks

        # Counters for logging
        self.hits = 0
        self.misses = 0

        super().__init__(
            bulk_load=self._bulk_lookup,
            batch_size=CACHE_BATCH_SIZE,
            coalesce_delay=CACHE_COALESCE_DELAY,
        )

    def _cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f'embed:{self._model}:{self._dimensions}:document:{text_hash}'

    async def _bulk_lookup(self, texts: Sequence[str]) -> Sequence[EmbedResponse]:
        """Look up cached embeddings, forward misses to embed loader.

        Redis errors propagate intentionally — Redis is required infrastructure
        (index state + embedding cache), not an optional optimization layer.
        """
        self._cache_tasks.check_health()

        t0 = time.perf_counter()
        keys = [self._cache_key(t) for t in texts]
        t_keys = time.perf_counter()

        cached = await self._redis.mget(keys)
        t_mget = time.perf_counter()

        # Separate hits and misses
        results: list[EmbedResponse | None] = []
        miss_indices: list[int] = []

        for i, raw in enumerate(cached):
            if raw is not None:
                values = np.frombuffer(raw, dtype=np.float32).tolist()
                # Skip Pydantic validation on cache hits — data was validated on write
                results.append(EmbedResponse.model_construct(values=values, dimensions=len(values)))
                self.hits += 1
            else:
                results.append(None)
                miss_indices.append(i)
        t_deser = time.perf_counter()

        self.misses += len(miss_indices)

        if miss_indices:
            # Forward misses to the embed loader (coalesces into API batches)
            miss_responses = await self._embed_loader.load_many([texts[i] for i in miss_indices])

            for idx, response in zip(miss_indices, miss_responses):
                results[idx] = response
                # Fire-and-forget cache write
                self._cache_tasks.submit(self._cache_write(texts[idx], response))

        t_end = time.perf_counter()
        hits = len(texts) - len(miss_indices)
        task_count = len(asyncio.all_tasks())
        logger.debug(
            f'[CACHE-DETAIL] n={len(texts)} hits={hits} '
            f'keys={(t_keys - t0) * 1000:.1f}ms '
            f'mget={(t_mget - t_keys) * 1000:.1f}ms '
            f'deser={(t_deser - t_mget) * 1000:.1f}ms '
            f'total={(t_end - t0) * 1000:.1f}ms '
            f'tasks={task_count}'
        )

        return results  # type: ignore[return-value]

    async def _cache_write(self, text: str, response: EmbedResponse) -> None:
        """Write embedding to cache with TTL."""
        key = self._cache_key(text)
        value = np.array(response.values, dtype=np.float32).tobytes()
        await self._redis.set(key, value, ex=CACHE_TTL_SECONDS)


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
