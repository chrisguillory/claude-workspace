"""Error boundary context managers for architectural exception handling.

Distinguishes boundary-level error handling (unexpected failures at architectural
edges) from business-logic error handling (expected failures in domain code).
Python's try/except conflates these — ErrorBoundary makes the intent explicit.

Three layers of error handling:

    Layer 1 — Business Logic:
        Local try/except for expected domain errors.
        ``try: find_user(id) except UserNotFound: return ErrorResponse(...)``

    Layer 2 — Scope Boundary:
        ErrorBoundary(exit_code=None) at request/task edges.
        Handles unexpected errors, continues serving.

    Layer 3 — Process Boundary:
        ErrorBoundary() at entry points.
        Handles unexpected errors, exits with non-zero code.

System exceptions (KeyboardInterrupt, SystemExit, GeneratorExit, CancelledError)
always pass through — they are not application errors. This uses Python's
designed hierarchy: ``issubclass(exc_type, Exception)`` is the positive check
that covers all application errors without brittle enumeration.

Cross-language equivalents:
    React <ErrorBoundary>, Rust panic::set_hook(), Elixir supervisor trees,
    Trio cancel scopes, Go defer+recover(), Java UncaughtExceptionHandler.

See also:
    - sys.excepthook: Global process-level hook, no scoping. Use when you
      want to customize ALL unhandled exception formatting without wrapping.
    - contextlib.suppress(): For expected exceptions to ignore silently.
      ErrorBoundary is for unexpected exceptions that need reporting.
"""

from __future__ import annotations

__all__ = ['ErrorBoundary']

import sys
import traceback
from collections.abc import Callable
from types import TracebackType
from typing import Self

type ErrorHandler = Callable[[Exception], None]


class ErrorBoundary:
    """Explicit error boundary for entry points and scope boundaries.

    Catches application exceptions (Exception subclasses) and delegates to a
    handler for reporting. System exceptions pass through unconditionally.

    Supports both ``with`` (sync) and ``async with`` (async) usage.

    Args:
        handler: Called with the caught exception for reporting/logging.
            Defaults to printing the traceback to stderr.
        exit_code: Process exit code after handling. ``None`` for scope
            boundaries that should suppress and continue. Defaults to 1
            (entry-point boundary).

    Examples:
        Process boundary (entry point, exits on error)::

            if __name__ == '__main__':
                with ErrorBoundary():
                    main()

        Scope boundary (request handler, suppress and continue)::

            for request in requests:
                with ErrorBoundary(exit_code=None, handler=log_error):
                    process_request(request)

        Async boundary::

            async with ErrorBoundary():
                await serve()

        Custom handler (Sentry, structured logging)::

            def send_to_sentry(exc: Exception) -> None:
                sentry_sdk.capture_exception(exc)

            with ErrorBoundary(handler=send_to_sentry):
                main()

    Composition:
        Error boundaries nest predictably. Inner boundaries handle first:

        - Inner with ``exit_code=1``: calls sys.exit(), outer sees SystemExit
          (BaseException, not Exception), passes through. Process exits.
        - Inner with ``exit_code=None``: suppresses, outer never sees it.
          Execution continues.
    """

    def __init__(
        self,
        *,
        handler: ErrorHandler | None = None,
        exit_code: int | None = 1,
    ) -> None:
        self._handler: ErrorHandler = handler or _default_handler
        self._exit_code = exit_code

    # -- Sync protocol --

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self._handle(exc_value)

    # -- Async protocol --

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self._handle(exc_value)

    # -- Core logic --

    def _handle(self, exc_value: BaseException | None) -> bool:
        """Core boundary logic — shared by sync and async paths.

        Returns True to suppress (scope boundary) or calls sys.exit()
        for process boundaries. Returns False for non-application exceptions.
        """
        if not isinstance(exc_value, Exception):
            return False  # No exception, or system exception — pass through

        self._handler(exc_value)

        if self._exit_code is not None:
            sys.exit(self._exit_code)

        return True  # Suppress at scope boundary


def _default_handler(exc: Exception) -> None:
    """Print exception with traceback to stderr."""
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
