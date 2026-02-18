"""Fire-and-forget async task tracking with deferred error propagation.

Submit coroutines without blocking. Errors are captured via done callbacks
and surfaced through explicit health checks or drain.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

__all__ = [
    'BackgroundTaskGroup',
]

logger = logging.getLogger(__name__)


class BackgroundTaskGroup:
    """Track background tasks with deferred error propagation.

    Tasks are submitted fire-and-forget via submit(). Errors are captured
    instantly via task done callbacks and raised on next check_health()
    or drain() call.

    Lifecycle: Create one per operation, not as a long-lived singleton.
    Once an error is captured, check_health() raises it on every subsequent
    call (fail-fast). The operation should fail and a fresh group should be
    created for the next operation.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._tasks: set[asyncio.Task[object]] = set()
        self._first_error: BaseException | None = None

    def submit(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Submit a coroutine for background execution.

        Returns immediately. Errors propagate via check_health() or drain().
        """
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task[object]) -> None:
        """Callback: capture first error, discard completed tasks."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and self._first_error is None:
            self._first_error = exc
            logger.error(f'[{self._name}] Background task failed: {exc}')

    def check_health(self) -> None:
        """Raise first captured error, if any. O(1) field read.

        Clears __traceback__ before re-raising to prevent unbounded frame
        accumulation (CPython issue #116862). Each `raise` prepends frames,
        and this method is called per-batch — without clearing, tracebacks
        grow by 2 frames per invocation. The original traceback was already
        logged at capture time in _on_done().
        """
        if self._first_error is not None:
            self._first_error.__traceback__ = None
            raise self._first_error

    async def drain(self) -> None:
        """Await all outstanding tasks. Raises first error if any failed."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self.check_health()

    def cancel_all(self) -> None:
        """Cancel all outstanding tasks."""
        for task in self._tasks:
            task.cancel()

    @property
    def pending_count(self) -> int:
        """Number of tasks still running."""
        return len(self._tasks)
