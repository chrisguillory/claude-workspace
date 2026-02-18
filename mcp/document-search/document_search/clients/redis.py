"""Async Redis client for embedding cache and index state.

Thin wrapper around redis.asyncio with connection pooling and concurrency control.
Binary mode (decode_responses=False) for efficient numpy embedding storage.
Hash operations support the IndexStateStore for per-file index state.
Port discovery handles Docker Compose ephemeral port mapping.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import redis.asyncio as aioredis

__all__ = [
    'RedisClient',
    'discover_redis_port',
]

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with connection pooling and concurrency control.

    Uses binary mode (decode_responses=False) for efficient numpy embedding
    storage. All operations are gated by a semaphore to prevent Redis
    saturation from fire-and-forget cache writes.

    Hash operations (hset, hgetall, hget) support the IndexStateStore
    for per-file index state tracking.
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 6379,
        *,
        max_connections: int = 50,
        max_concurrent: int = 20,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 2.0,
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

    async def ping(self) -> bool:
        """Verify Redis connectivity."""
        return await self._client.ping()

    # --- Key/value operations (embedding cache) ---

    async def mget(self, keys: Sequence[str]) -> Sequence[bytes | None]:
        """Batch get, gated by concurrency semaphore."""
        async with self._semaphore:
            return await self._client.mget(keys)

    async def set(self, name: str, value: bytes, *, ex: int | None = None) -> bool | None:
        """Set with optional TTL, gated by concurrency semaphore."""
        async with self._semaphore:
            return await self._client.set(name, value, ex=ex)

    # --- Hash operations (index state) ---

    async def hset(self, name: str, mapping: Mapping[str, str | bytes]) -> int:
        """Set multiple hash fields atomically.

        String values are UTF-8 encoded for binary-mode compatibility.
        """
        encoded = {k.encode(): (v if isinstance(v, bytes) else v.encode()) for k, v in mapping.items()}
        async with self._semaphore:
            return await self._client.hset(name, mapping=encoded)

    async def hgetall(self, name: str) -> Mapping[bytes, bytes]:
        """Get all fields and values from a hash."""
        async with self._semaphore:
            return await self._client.hgetall(name)

    async def hget(self, name: str, field: str) -> bytes | None:
        """Get a single hash field value."""
        async with self._semaphore:
            return await self._client.hget(name, field.encode())

    # --- Key management ---

    async def delete(self, *names: str) -> int:
        """Delete one or more keys."""
        async with self._semaphore:
            return await self._client.delete(*names)

    async def scan_iter(self, *, match: str, count: int = 100) -> AsyncIterator[str]:
        """Iterate keys matching glob pattern. Decodes bytes to strings.

        SCAN is cursor-based and non-blocking. The semaphore is NOT held
        across the full iteration — each cursor step acquires independently.
        """
        async for key in self._client.scan_iter(match=match, count=count):
            yield key.decode()

    def pipeline(self) -> aioredis.Pipeline:
        """Get a pipeline for batching multiple commands in one round-trip.

        Caller is responsible for executing the pipeline. The semaphore is
        NOT held — pipelines manage their own connection lifecycle.
        """
        return self._client.pipeline(transaction=False)

    async def close(self) -> None:
        """Close connection pool."""
        await self._client.aclose()


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
