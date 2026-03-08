"""Concurrency tracking for async operations.

Provides lightweight instrumentation for measuring concurrent API calls,
latency, and throughput without cluttering call sites.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

__all__ = ['ConcurrencyTracker']


class ConcurrencyTracker:
    """Track concurrent operations with minimal call-site overhead.

    Usage:
        tracker = ConcurrencyTracker("GEMINI", log_interval=5.0)

        async with tracker.track():
            await api_call()

        # Stats are automatically logged every 5 seconds
        # At shutdown, call tracker.stop() to log final stats
    """

    def __init__(
        self,
        name: str,
        logger: logging.Logger | None = None,
        log_interval: float | None = 5.0,
    ) -> None:
        """Initialize tracker.

        Args:
            name: Identifier for logging (e.g., "GEMINI", "QDRANT").
            logger: Logger instance. Defaults to module logger.
            log_interval: Seconds between periodic stats logs. None to disable.
        """
        self.name = name
        self._logger = logger or logging.getLogger(__name__)
        self._log_interval = log_interval
        self._in_flight = 0
        self._max_in_flight = 0
        self._total_calls = 0
        self._total_time = 0.0
        self._monitor_task: asyncio.Task[None] | None = None
        self._last_logged_calls = 0

    def _start_monitor(self) -> None:
        """Start background monitor if configured and not already running."""
        if self._log_interval and self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self) -> None:
        """Periodically log stats while there's activity."""
        while True:
            await asyncio.sleep(self._log_interval or 5.0)
            # Only log if there's been activity since last log
            if self._total_calls > self._last_logged_calls:
                self._logger.debug(f'[{self.name}] {self.stats}')
                self._last_logged_calls = self._total_calls

    @asynccontextmanager
    async def track(self) -> AsyncIterator[None]:
        """Track a single operation.

        Measures:
        - Concurrent operations (current and peak)
        - Total call count
        - Total and average latency
        """
        # Start monitor on first tracked operation
        if self._total_calls == 0:
            self._start_monitor()

        self._in_flight += 1
        self._max_in_flight = max(self._max_in_flight, self._in_flight)
        self._total_calls += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._total_time += time.perf_counter() - t0
            self._in_flight -= 1

    @property
    def stats(self) -> dict[str, Any]:
        """Current statistics."""
        return {
            'total_calls': self._total_calls,
            'max_concurrent': self._max_in_flight,
            'in_flight': self._in_flight,
            'total_time_s': round(self._total_time, 2),
            'avg_latency_ms': round(self._total_time / self._total_calls * 1000, 1) if self._total_calls else 0,
        }

    @property
    def in_flight(self) -> int:
        """Current number of in-flight operations."""
        return self._in_flight

    def log_summary(self) -> None:
        """Log final statistics summary."""
        self._logger.debug(f'[{self.name}] FINAL {self.stats}')

    def stop(self) -> None:
        """Stop background monitor and log final stats."""
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self._total_calls > 0:
            self.log_summary()

    def reset(self) -> None:
        """Reset all counters."""
        self._in_flight = 0
        self._max_in_flight = 0
        self._total_calls = 0
        self._total_time = 0.0
        self._last_logged_calls = 0
