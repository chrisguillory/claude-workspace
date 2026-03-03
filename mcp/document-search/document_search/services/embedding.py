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
        model: str,
        dimensions: int,
    ) -> None:
        """Initialize service.

        Args:
            client: Embedding client.
            batch_size: Max texts per API batch.
            coalesce_delay: Seconds to wait for request coalescing (default 10ms).
            redis: Redis client for embedding cache.
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

    # ── Reverse index pass-through ────────────────────────────

    async def update_file_cache_index(self, file_path: str, texts: Sequence[str]) -> None:
        """Update reverse index mapping file → embedding cache keys."""
        await self._cache_loader.update_file_index(file_path, texts)

    async def submit_file_cache_index(self, file_path: str, texts: Sequence[str]) -> None:
        """Fire-and-forget reverse index update via write batcher."""
        await self._cache_loader.submit_file_index(file_path, texts)

    async def refresh_file_cache_index_ttl(self, file_path: str) -> None:
        """Refresh TTL on reverse index for a chunk-cached file."""
        await self._cache_loader.refresh_file_index_ttl(file_path)

    async def invalidate_file_cache(self, file_path: str) -> int:
        """Delete cached embeddings for a file via reverse index."""
        return await self._cache_loader.invalidate_file_cache(file_path)

    async def invalidate_path_cache(self, path_prefix: str) -> int:
        """Delete cached embeddings for files under a directory."""
        return await self._cache_loader.invalidate_path_prefix(path_prefix)

    # ── Lifecycle ──────────────────────────────────────────────

    async def drain_writes(self) -> None:
        """Flush all pending cache and index writes."""
        await self._cache_loader.drain_writes()

    def check_write_health(self) -> None:
        """Raise first error from fire-and-forget write operations."""
        self._cache_loader.check_write_health()

    def cancel_writes(self) -> None:
        """Cancel in-flight write batchers."""
        self._cache_loader.cancel_writes()


class CacheLoader(GenericBatchLoader[str, EmbedResponse]):
    """Batch cache lookups with miss forwarding.

    Coalesces concurrent cache lookups into Redis MGET batches.
    Cache hits return immediately. Misses are forwarded to the
    embed loader for coalescing into API batches.

    Cache and index writes are batched via separate GenericBatchLoader
    instances using submit/drain for fire-and-forget semantics.

    Cache key format: embed:{model}:{dims}:document:{sha256[:16]}
    Intent is always 'document' since only embed_text() uses this path.
    """

    def __init__(
        self,
        redis: RedisClient,
        model: str,
        dimensions: int,
        embed_loader: _EmbedLoader,
    ) -> None:
        self._redis = redis
        self._model = model
        self._dimensions = dimensions
        self._embed_loader = embed_loader

        # Counters for logging
        self.hits = 0
        self.misses = 0

        # Write batchers for fire-and-forget cache/index writes
        self._cache_write_batcher = GenericBatchLoader[tuple[str, bytes, int], None](
            bulk_load=self._bulk_cache_write,
            batch_size=200,
            coalesce_delay=0.050,  # 50ms — writes are less latency-sensitive
        )
        self._index_write_batcher = GenericBatchLoader[tuple[str, tuple[bytes, ...]], None](
            bulk_load=self._bulk_index_write,
            batch_size=100,
            coalesce_delay=0.050,
        )

        super().__init__(
            bulk_load=self._bulk_lookup,
            batch_size=CACHE_BATCH_SIZE,
            coalesce_delay=CACHE_COALESCE_DELAY,
        )

    def cache_key(self, text: str) -> str:
        """Generate cache key for a text.

        Key format: embed:{model}:{dims}:document:{sha256[:16]}
        Content-addressed for cross-file deduplication.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f'embed:{self._model}:{self._dimensions}:document:{text_hash}'

    # ── Reverse index: file path → cache keys ───────────────────

    async def submit_file_index(self, file_path: str, texts: Sequence[str]) -> None:
        """Fire-and-forget reverse index update via batcher."""
        index_key = self._index_key(file_path)
        cache_keys = tuple(self.cache_key(t).encode() for t in texts) if texts else ()
        await self._index_write_batcher.submit((index_key, cache_keys))

    async def refresh_file_index_ttl(self, file_path: str) -> None:
        """Refresh TTL on an existing reverse index entry (no content change)."""
        index_key = self._index_key(file_path)
        await self._redis.expire(index_key, CACHE_TTL_SECONDS)

    async def update_file_index(self, file_path: str, texts: Sequence[str]) -> None:
        """Atomically replace reverse index mapping file → cache keys.

        Uses MULTI/EXEC transaction: UNLINK old set → SADD new keys → EXPIRE.
        """
        index_key = self._index_key(file_path)
        if not texts:
            # No embeddings to track — remove stale reverse index
            await self._redis.unlink(index_key)
            return
        cache_keys = [self.cache_key(t) for t in texts]
        pipe = self._redis.transaction()
        pipe.unlink(index_key)
        pipe.sadd(index_key, *[k.encode() for k in cache_keys])
        pipe.expire(index_key, CACHE_TTL_SECONDS)
        await self._redis.execute_pipeline(pipe)

    async def invalidate_file_cache(self, file_path: str) -> int:
        """Delete all cached embeddings for a file via reverse index."""
        index_key = self._index_key(file_path)
        cache_keys = await self._redis.smembers(index_key)
        if cache_keys:
            await self._redis.unlink(*cache_keys, index_key)
        return len(cache_keys)

    async def invalidate_path_prefix(self, path_prefix: str) -> int:
        """Delete all cached embeddings for files under a directory."""
        # Ensure directory boundary to avoid /dir1* matching /dir10/
        if path_prefix and not path_prefix.endswith('/'):
            path_prefix += '/'
        pattern = f'embed-idx:file:{path_prefix}*'
        total = 0
        async for index_key in self._redis.scan_iter(match=pattern):
            cache_keys = await self._redis.smembers(index_key)
            if cache_keys:
                await self._redis.unlink(*cache_keys, index_key)
                total += len(cache_keys)
        return total

    # ── Lifecycle ──────────────────────────────────────────────

    async def drain_writes(self) -> None:
        """Flush all pending cache and index writes.

        Both batchers always drain via try/finally. If both raise, the index
        error propagates with cache error chained as __context__ (Python
        auto-chaining when exception occurs during exception handling).
        """
        try:
            await self._cache_write_batcher.drain()
        finally:
            await self._index_write_batcher.drain()

    def check_write_health(self) -> None:
        """Raise first error from fire-and-forget write operations."""
        self._cache_write_batcher.check_health()
        self._index_write_batcher.check_health()

    def cancel_writes(self) -> None:
        """Cancel in-flight write batchers."""
        self._cache_write_batcher.cancel_all()
        self._index_write_batcher.cancel_all()

    # ── Private: batch loading, cache writes, helpers ──────────────

    @staticmethod
    def _index_key(file_path: str) -> str:
        """Reverse index key for a file path."""
        return f'embed-idx:file:{file_path}'

    async def _bulk_lookup(self, texts: Sequence[str]) -> Sequence[EmbedResponse]:
        """Look up cached embeddings, forward misses to embed loader.

        Redis errors propagate intentionally — Redis is required infrastructure
        (index state + embedding cache), not an optional optimization layer.
        """
        self._cache_write_batcher.check_health()
        self._index_write_batcher.check_health()

        t0 = time.perf_counter()
        keys = [self.cache_key(t) for t in texts]
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

            # Prepare cache writes for batching
            write_items: list[tuple[str, bytes, int]] = []
            for idx, response in zip(miss_indices, miss_responses):
                results[idx] = response
                key = self.cache_key(texts[idx])
                value = np.array(response.values, dtype=np.float32).tobytes()
                write_items.append((key, value, CACHE_TTL_SECONDS))

            # Fire-and-forget cache writes via batcher
            await self._cache_write_batcher.submit_many(write_items)

        t_end = time.perf_counter()
        hits = len(texts) - len(miss_indices)
        if logger.isEnabledFor(logging.DEBUG):
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

    async def _bulk_cache_write(self, items: Sequence[tuple[str, bytes, int]]) -> Sequence[None]:
        """Pipeline cache SET operations."""
        pipe = self._redis.pipeline()
        for key, value, ttl in items:
            pipe.set(key, value, ex=ttl)
        await self._redis.execute_pipeline(pipe)
        return [None] * len(items)

    async def _bulk_index_write(self, items: Sequence[tuple[str, tuple[bytes, ...]]]) -> Sequence[None]:
        """Pipeline reverse index updates (UNLINK + SADD + EXPIRE).

        Non-transactional pipeline (vs transaction() in update_file_index) —
        trades per-file atomicity for throughput. Reverse index is best-effort;
        partial writes self-heal on next re-index.
        """
        pipe = self._redis.pipeline()
        for index_key, cache_keys in items:
            if cache_keys:
                pipe.unlink(index_key)
                pipe.sadd(index_key, *cache_keys)
                pipe.expire(index_key, CACHE_TTL_SECONDS)
            else:
                pipe.unlink(index_key)
        await self._redis.execute_pipeline(pipe)
        return [None] * len(items)


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
