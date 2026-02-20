"""Error boundary with type-based handler dispatch.

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

Patterns:

    Type-based dispatch (like FastAPI's ``@app.exception_handler``)::

        boundary = ErrorBoundary(exit_code=None)

        @boundary.handler(ValidationError)
        def handle_validation(exc: ValidationError) -> None:
            show_field_errors(exc)

        @boundary.handler(Exception)
        def handle_crash(exc: Exception) -> None:
            show_traceback(exc)

        @boundary
        def main() -> None:
            ...

    Exception enrichment (carry context from raise site to handler)::

        class RichValidationError(Exception):
            def __init__(self, error: ValidationError, raw: dict):
                self.error = error
                self.raw = raw
                super().__init__(str(error))

        try:
            data = validate(raw_input)
        except ValidationError as e:
            raise RichValidationError(e, raw_input) from e

    See ``scripts/statusline.py`` for the canonical real-world example.

Cross-language equivalents:
    React <ErrorBoundary>, Rust panic::set_hook(), Elixir supervisor trees,
    Trio cancel scopes, Go defer+recover(), Java UncaughtExceptionHandler.

See also:
    - ``functools.singledispatch``: Powers the type-based handler dispatch
      internally. Handlers are matched by MRO.
    - ``sys.excepthook``: Global process-level hook, no scoping. Use as a
      fallback for errors before ErrorBoundary loads (e.g., import failures).
    - ``contextlib.suppress()``: For expected exceptions to ignore silently.
      ErrorBoundary is for unexpected exceptions that need reporting.
"""

from __future__ import annotations

__all__ = [
    'ErrorBoundary',
    'ErrorHandler',
]

import asyncio
import functools
import sys
import traceback
from collections.abc import Callable
from functools import singledispatch
from types import TracebackType
from typing import Any, Self, TypeVar, cast

type ErrorHandler = Callable[[Exception], None]

_F = TypeVar('_F', bound=Callable[..., object])


class ErrorBoundary:
    """Error boundary with optional type-based handler dispatch.

    Catches application exceptions (Exception subclasses) and delegates to
    registered handlers. System exceptions pass through unconditionally.

    Supports three usage forms:

    - **Decorator**: ``@boundary`` on sync or async functions
    - **Context manager**: ``with boundary:`` or ``async with boundary:``
    - **Simple handler**: ``ErrorBoundary(handler=func)`` for single-function handling

    Type dispatch uses ``functools.singledispatch`` internally — handlers are
    matched by MRO, so registering for ``Exception`` acts as a catch-all.

    Args:
        handler: Convenience for registering a catch-all handler (equivalent
            to ``@boundary.handler(Exception)``). Defaults to printing
            the traceback to stderr.
        exit_code: Process exit code after handling. ``None`` for scope
            boundaries that should suppress and continue. Defaults to 1
            (entry-point boundary).

    Examples:
        Entry point with typed handlers (decorator form)::

            boundary = ErrorBoundary(exit_code=None)

            @boundary.handler(ValidationError)
            def handle_validation(exc: ValidationError) -> None:
                show_field_errors(exc)

            @boundary.handler(Exception)
            def handle_crash(exc: Exception) -> None:
                show_traceback(exc)

            @boundary
            def main() -> None:
                ...

            if __name__ == '__main__':
                main()

        Entry point with simple handler::

            @ErrorBoundary(handler=log_error)
            def main() -> None:
                ...

        Scope boundary (context manager, suppress and continue)::

            for request in requests:
                with ErrorBoundary(exit_code=None, handler=log_error):
                    process_request(request)

        Async entry point (auto-detected)::

            @ErrorBoundary(handler=log_error)
            async def serve() -> None:
                await start_server()

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
        self._dispatch = singledispatch(_default_handler)
        if handler is not None:
            self._dispatch.register(Exception, handler)
        self._exit_code = exit_code

    def handler(self, exc_type: type[Exception]) -> Callable[[Callable[..., None]], Callable[..., None]]:
        """Register a handler for a specific exception type.

        Uses ``functools.singledispatch`` for MRO-based matching — registering
        for ``Exception`` acts as a catch-all default.

        Example::

            @boundary.handler(ValidationError)
            def handle_validation(exc: ValidationError) -> None:
                ...
        """
        return self._dispatch.register(exc_type)

    # -- Decorator protocol --

    def __call__(self, func: _F) -> _F:
        """Decorate a function with this error boundary.

        Auto-detects sync vs async and wraps accordingly. Parens required::

            @ErrorBoundary()          # correct
            @ErrorBoundary(handler=f) # correct
            @ErrorBoundary            # TypeError — parens required
        """
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self:
                    return await func(*args, **kwargs)

            return cast(_F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return cast(_F, sync_wrapper)

    # -- Sync context manager --

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return self._handle(exc_value)

    # -- Async context manager --

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
        """Core boundary logic — shared by all protocol paths.

        Returns True to suppress (scope boundary) or calls sys.exit()
        for process boundaries. Returns False for non-application exceptions.

        Handler failures cannot breach the boundary — if the registered handler
        raises, we fall back to stderr reporting of the original exception.
        If that also fails (e.g. stderr broken), we proceed silently with
        the configured suppress/exit behavior.
        """
        if not isinstance(exc_value, Exception):
            return False  # No exception, or system exception — pass through

        try:
            self._dispatch(exc_value)
        except Exception:
            try:  # noqa: SIM105 — explicit try/except/pass preserves comment explaining why
                _default_handler(exc_value)
            except Exception:
                pass  # Both handlers failed (e.g. stderr broken); proceed with exit/suppress

        if self._exit_code is not None:
            sys.exit(self._exit_code)

        return True  # Suppress at scope boundary


def _default_handler(exc: Exception) -> None:
    """Print exception with traceback to stderr."""
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
