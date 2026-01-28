"""Type stubs for pyrate_limiter library (v4.0.2)."""

from collections.abc import Sequence
from enum import Enum

class Duration(int, Enum):
    SECOND = 1000
    MINUTE = 60000
    HOUR = 3600000
    DAY = 86400000
    WEEK = 604800000

class Rate:
    def __init__(self, limit: int, interval: int | Duration) -> None: ...

class Limiter:
    def __init__(
        self,
        argument: Rate | Sequence[Rate],
        buffer_ms: int = 50,
    ) -> None: ...
    async def try_acquire_async(
        self,
        name: str = 'pyrate',
        weight: int = 1,
        blocking: bool = True,
        timeout: int = -1,
    ) -> bool: ...
