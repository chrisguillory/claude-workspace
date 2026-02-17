"""Type stubs for redis.asyncio - async Redis client."""

from collections.abc import Sequence

class ConnectionPool:
    def __init__(
        self,
        host: str = ...,
        port: int = ...,
        *,
        max_connections: int = ...,
        socket_timeout: float = ...,
        socket_connect_timeout: float = ...,
        decode_responses: bool = ...,
    ) -> None: ...

class Redis:
    def __init__(self, *, connection_pool: ConnectionPool) -> None: ...
    async def ping(self) -> bool: ...
    async def mget(self, keys: Sequence[str]) -> list[bytes | None]: ...
    async def set(self, name: str, value: str | bytes, *, ex: int | None = ...) -> bool | None: ...
    async def aclose(self) -> None: ...
