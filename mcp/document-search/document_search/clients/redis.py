"""Async Redis client for embedding cache.

Thin wrapper around redis.asyncio with connection pooling and concurrency control.
Binary mode (decode_responses=False) for efficient numpy embedding storage.
Port discovery handles Docker Compose ephemeral port mapping.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Sequence
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

    async def mget(self, keys: Sequence[str]) -> Sequence[bytes | None]:
        """Batch get, gated by concurrency semaphore."""
        async with self._semaphore:
            return await self._client.mget(keys)

    async def set(self, name: str, value: bytes, *, ex: int | None = None) -> bool | None:
        """Set with optional TTL, gated by concurrency semaphore."""
        async with self._semaphore:
            return await self._client.set(name, value, ex=ex)

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
