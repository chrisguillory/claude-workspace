"""Circuit breaker with open/close transition logging.

Private module - import from _retry package.
"""

from __future__ import annotations

import logging
from types import TracebackType

import circuitbreaker

__all__ = [
    'LoggingCircuitBreaker',
]

logger = logging.getLogger(__name__)


class LoggingCircuitBreaker(circuitbreaker.CircuitBreaker):
    """CircuitBreaker that logs at the open and close transitions.

    circuitbreaker 2.1.3 exposes no on_open/on_close callbacks, so transitions
    are observed by hooking __exit__ (the open edge) and reset (the close edge).

    __call_failed — where the open transition actually fires — is name-mangled
    (_CircuitBreaker__call_failed); a subclass override is a silent no-op, so the
    open edge is detected by comparing the public `opened` flag across super().__exit__.
    """

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, _traceback: TracebackType | None
    ) -> bool:
        was_open = self.opened
        result = super().__exit__(exc_type, exc_value, _traceback)
        if not was_open and self.opened:
            logger.error(
                '[CIRCUIT] %s breaker OPENED after %s consecutive failures (last: %s)',
                self.name,
                self.failure_count,
                self.last_failure,
            )
        return result

    def reset(self) -> None:
        was_open = not self.closed
        super().reset()
        if was_open:
            logger.info('[CIRCUIT] %s breaker CLOSED — recovered', self.name)
