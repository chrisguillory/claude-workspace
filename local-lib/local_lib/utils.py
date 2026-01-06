"""Shared utilities for MCP servers."""

from __future__ import annotations

# Standard Library
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Third-Party Libraries
from mcp.server.fastmcp import Context


def encode_project_path(path: Path | str) -> str:
    """Encode filesystem path for Claude Code's directory naming convention.

    Claude Code stores project data in ~/.claude/projects/{encoded_path}/ where
    the path is encoded by replacing certain characters with hyphens.

    This encoding is LOSSY and non-reversible (5 different chars -> 1 char).
    To recover the original path, read the 'cwd' field from session records.

    Args:
        path: Filesystem path to encode (e.g., /Users/chris/My Project)

    Returns:
        Encoded string for use as directory name in ~/.claude/projects/

    Examples:
        >>> encode_project_path("/Users/chris/project")
        '-Users-chris-project'

        >>> encode_project_path("/Users/chris/Mobile Documents/com~apple~CloudDocs")
        '-Users-chris-Mobile-Documents-com-apple-CloudDocs'

        >>> encode_project_path("/Users/chris/my_project.app")
        '-Users-chris-my-project-app'
    """
    result = str(path) if isinstance(path, Path) else path
    # Claude Code encodes paths by replacing these characters with hyphens:
    for char in ['/', '.', ' ', '~', '_']:
        result = result.replace(char, '-')
    return result


class DualLogger:
    """Logs messages to both stdout and MCP client context."""

    def __init__(self, ctx: Context[Any, Any, Any]):
        self.ctx = ctx

    def _timestamp(self) -> str:
        return datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')

    async def info(self, msg: str) -> None:
        print(f'[{self._timestamp()}] [INFO] {msg}')
        await self.ctx.info(msg)

    async def debug(self, msg: str) -> None:
        print(f'[{self._timestamp()}] [DEBUG] {msg}')
        await self.ctx.debug(msg)

    async def warning(self, msg: str) -> None:
        print(f'[{self._timestamp()}] [WARNING] {msg}')
        await self.ctx.warning(msg)

    async def error(self, msg: str) -> None:
        print(f'[{self._timestamp()}] [ERROR] {msg}')
        await self.ctx.error(msg)


class Timer:
    """Simple stopwatch-style timer for measuring elapsed time."""

    def __init__(self) -> None:
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
        ('d', 86400),  # days
        ('hr', 3600),  # hours
        ('min', 60),  # minutes
        ('sec', 1),  # seconds
    ]

    for unit, count in intervals:
        if seconds >= count:
            value = seconds / count
            value_str = f'{value:.1f}'.rstrip('0').rstrip('.')
            return f'{value_str} {unit}'

    return '0 sec'
