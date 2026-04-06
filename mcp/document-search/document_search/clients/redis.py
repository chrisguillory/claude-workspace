"""Async Redis client for embedding cache and index state.

Thin wrapper around redis.asyncio with connection pooling and concurrency control.
Binary mode (decode_responses=False) for efficient numpy embedding storage.
Hash operations support the IndexStateStore for per-file index state.
Port discovery handles Docker Compose ephemeral port mapping.

Connection governance:
- Single-command ops (mget, set, hset, etc.) gated by `_semaphore`
- Multi-command ops (pipeline, transaction) gated by `_pipe_semaphore`
- scan_iter bypasses both (cursor-based, redis-py manages connection)
- High-water mark tracks peak concurrent connections for diagnostics
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence, Set
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis

__all__ = [
    'ConnectionStats',
    'RedisClient',
    'discover_redis_port',
]

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and concurrency control.

    Uses binary mode (decode_responses=False) for efficient numpy embedding
    storage. Two semaphores govern connection usage:

    - _semaphore: single-command ops (mget, set, hset, etc.)
    - _pipe_semaphore: multi-command ops (pipeline, transaction)

    Both draw from the same connection pool. Peak usage is tracked via
    high-water mark counters for diagnostics.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 6379,
        *,
        max_connections: int = 5000,  # Lazy ceiling. Semaphores (500+500) provide backpressure. Server maxclients: 65,536.
        max_concurrent: int = 500,
        max_concurrent_pipelines: int = 500,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 5.0,
    ) -> None:
        pool = aioredis.ConnectionPool(
            host=host,
            port=port,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,
        )
        self._client = aioredis.Redis(connection_pool=pool)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pipe_semaphore = asyncio.Semaphore(max_concurrent_pipelines)

        # Diagnostics: track peak concurrent connections
        self._max_connections = max_connections
        self._active_single = 0
        self._active_pipeline = 0
        self._hwm_single = 0
        self._hwm_pipeline = 0
        self._hwm_total = 0

    async def ping(self) -> bool:
        """Verify Redis connectivity."""
        return await self._client.ping()

    @property
    def connection_stats(self) -> ConnectionStats:
        """Point-in-time connection diagnostics.

        Returns current and peak (high-water mark) connection counts.
        """
        return ConnectionStats(
            active_single=self._active_single,
            active_pipeline=self._active_pipeline,
            active_total=self._active_single + self._active_pipeline,
            hwm_single=self._hwm_single,
            hwm_pipeline=self._hwm_pipeline,
            hwm_total=self._hwm_total,
            pool_max=self._max_connections,
        )

    def reset_hwm(self) -> None:
        """Reset high-water marks. Call at the start of each indexing operation."""
        self._hwm_single = 0
        self._hwm_pipeline = 0
        self._hwm_total = 0

    # --- Key/value operations (embedding cache) ---

    async def mget(self, keys: Sequence[str]) -> Sequence[bytes | None]:
        """Batch get, gated by concurrency semaphore."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.mget(keys)
            finally:
                self._track_single(-1)

    async def set(self, name: str, value: bytes, *, ex: int | None = None) -> bool | None:
        """Set with optional TTL, gated by concurrency semaphore."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.set(name, value, ex=ex)
            finally:
                self._track_single(-1)

    # --- Hash operations (index state) ---

    async def hset(self, name: str, mapping: Mapping[str, str | bytes]) -> int:
        """Set multiple hash fields atomically.

        String values are UTF-8 encoded for binary-mode compatibility.
        """
        encoded = {k.encode(): (v if isinstance(v, bytes) else v.encode()) for k, v in mapping.items()}
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.hset(name, mapping=encoded)
            finally:
                self._track_single(-1)

    async def hgetall(self, name: str) -> Mapping[bytes, bytes]:
        """Get all fields and values from a hash."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.hgetall(name)
            finally:
                self._track_single(-1)

    async def hget(self, name: str, field: str) -> bytes | None:
        """Get a single hash field value."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.hget(name, field.encode())
            finally:
                self._track_single(-1)

    # --- Key management ---

    async def delete(self, *names: str) -> int:
        """Delete one or more keys."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.delete(*names)
            finally:
                self._track_single(-1)

    async def unlink(self, *names: str) -> int:
        """Unlink one or more keys (non-blocking memory deallocation).

        Preferred over delete() for Sets and large values — O(1) on the
        main thread, defers memory freeing to a background thread.
        """
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.unlink(*names)
            finally:
                self._track_single(-1)

    async def expire(self, name: str, time: int) -> bool:
        """Set TTL on a key. Returns True if key exists and TTL was set."""
        async with self._semaphore:
            self._track_single(1)
            try:
                return await self._client.expire(name, time)
            finally:
                self._track_single(-1)

    async def smembers(self, name: str) -> Set[str]:
        """Get all members of a set, decoded to strings."""
        async with self._semaphore:
            self._track_single(1)
            try:
                raw = await self._client.smembers(name)
                return {m.decode() for m in raw}
            finally:
                self._track_single(-1)

    async def scan_iter(self, *, match: str, count: int = 100) -> AsyncIterator[str]:
        """Iterate keys matching glob pattern. Decodes bytes to strings.

        SCAN is cursor-based and non-blocking. Bypasses semaphores —
        redis-py manages its own connection for scan iteration.
        """
        async for key in self._client.scan_iter(match=match, count=count):
            yield key.decode()

    async def execute_pipeline(  # strict_typing_linter.py: loose-typing — redis pipeline returns heterogeneous command results
        self,
        pipe: aioredis.Pipeline,
    ) -> Sequence[Any]:
        """Execute a pipeline, gated by pipeline semaphore.

        Ensures pipeline/transaction connections are bounded.
        Tracks connection usage for diagnostics.
        """
        async with self._pipe_semaphore:
            self._track_pipeline(1)
            try:
                return await pipe.execute()
            finally:
                self._track_pipeline(-1)

    def pipeline(self) -> aioredis.Pipeline:
        """Get a pipeline for batching multiple commands in one round-trip.

        IMPORTANT: Use execute_pipeline() to run — it gates through the
        pipeline semaphore and tracks connection usage.
        """
        return self._client.pipeline(transaction=False)

    def transaction(self) -> aioredis.Pipeline:
        """Get a transactional pipeline (MULTI/EXEC) for atomic operations.

        IMPORTANT: Use execute_pipeline() to run — it gates through the
        pipeline semaphore and tracks connection usage.
        """
        return self._client.pipeline(transaction=True)

    async def close(self) -> None:
        """Close connection pool."""
        await self._client.aclose()

    # --- Private: connection tracking ---

    def _track_single(self, delta: int) -> None:
        self._active_single += delta
        if delta > 0:
            total = self._active_single + self._active_pipeline
            self._hwm_single = max(self._hwm_single, self._active_single)
            self._hwm_total = max(self._hwm_total, total)

    def _track_pipeline(self, delta: int) -> None:
        self._active_pipeline += delta
        if delta > 0:
            total = self._active_single + self._active_pipeline
            self._hwm_pipeline = max(self._hwm_pipeline, self._active_pipeline)
            self._hwm_total = max(self._hwm_total, total)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ConnectionStats:
    """Point-in-time Redis connection diagnostics."""

    active_single: int
    active_pipeline: int
    active_total: int
    hwm_single: int
    hwm_pipeline: int
    hwm_total: int
    pool_max: int


def discover_redis_port(compose_dir: Path) -> int:
    """Discover ephemeral host port for Redis via Docker Compose.

    Args:
        compose_dir: Directory containing docker-compose.yaml.

    Returns:
        Host port mapped to Redis container port 6379.

    Raises:
        RuntimeError: If port discovery fails.
    """
    result = subprocess.run(
        ['docker', 'compose', 'port', 'redis', '6379'],
        check=False,
        capture_output=True,
        text=True,
        cwd=compose_dir,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Redis port discovery failed: {result.stderr.strip()}')

    # Output format: "0.0.0.0:12345"
    _, _, port_str = result.stdout.strip().rpartition(':')
    return int(port_str)
