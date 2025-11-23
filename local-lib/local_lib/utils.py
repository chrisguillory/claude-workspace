"""Shared utilities for MCP servers."""

from __future__ import annotations

# Standard Library
import time
from datetime import datetime

# Third-Party Libraries
from mcp.server.fastmcp import Context


class DualLogger:
    """Logs messages to both stdout and MCP client context."""

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def info(self, msg: str):
        print(f"[{self._timestamp()}] [INFO] {msg}")
        await self.ctx.info(msg)

    async def debug(self, msg: str):
        print(f"[{self._timestamp()}] [DEBUG] {msg}")
        await self.ctx.debug(msg)

    async def warning(self, msg: str):
        print(f"[{self._timestamp()}] [WARNING] {msg}")
        await self.ctx.warning(msg)

    async def error(self, msg: str):
        print(f"[{self._timestamp()}] [ERROR] {msg}")
        await self.ctx.error(msg)


class Timer:
    """Simple stopwatch-style timer for measuring elapsed time."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.perf_counter() - self._start

    def elapsed_ms(self) -> int:
        """Return elapsed time in milliseconds."""
        return int(self.elapsed() * 1000)


def humanize_seconds(seconds: float) -> str:
    """Convert seconds to human-readable duration with abbreviated units.

    Uses terse but complete format following Unix conventions:
    - 45 sec
    - 1.5 min
    - 2.5 hr
    - 3d

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string with abbreviated unit
    """
    intervals = [
        ('d', 86400),    # days
        ('hr', 3600),    # hours
        ('min', 60),     # minutes
        ('sec', 1)       # seconds
    ]

    for unit, count in intervals:
        if seconds >= count:
            value = seconds / count
            value_str = f"{value:.1f}".rstrip('0').rstrip('.')
            return f"{value_str} {unit}"

    return "0 sec"
