"""Error boundary with type-based handler dispatch.

Distinguishes boundary-level error handling (unexpected failures at architectural
edges) from business-logic error handling (expected failures in domain code).
Python's try/except conflates these — ErrorBoundary makes the intent explicit.

Four layers of error handling:

    Layer 1 — Business Logic:
        Local try/except for expected domain errors.
        ``try: find_user(id) except UserNotFound: return ErrorResponse(...)``

    Layer 2 — Library Boundary:
        Translate third-party exceptions (``library_boundary.py``).
        ``with boundary: lib.parse(cmd)``

    Layer 3 — Scope Boundary:
        ErrorBoundary() at request/task edges.
        Handles unexpected errors, continues serving.

    Layer 4 — Process Boundary:
        ErrorBoundary(exit_code=N) at entry points.
        Handles expected and unexpected errors, exits with non-zero code.
        For Claude Code hooks, use exit_code=2 — exit 1 is a black hole
        where the model sees nothing (see hooks/run-linter.py docstring).

System exceptions (KeyboardInterrupt, SystemExit, GeneratorExit, CancelledError)
always pass through — they are not application errors. This uses Python's
designed hierarchy: ``issubclass(exc_type, Exception)`` is the positive check
that covers all application errors without brittle enumeration.

Patterns:

    Type-based dispatch with fallback return values (HTTP handlers)::

        boundary = ErrorBoundary()

        @boundary.handler(ValidationError)
        def handle_validation(exc: ValidationError) -> JSONResponse:
            return JSONResponse({'error': str(exc)}, status_code=422)

        @boundary.handler(Exception)
        def handle_crash(exc: Exception) -> JSONResponse:
            return JSONResponse({'error': 'Internal error'}, status_code=500)

        @boundary
        async def handle_request(request: Request) -> JSONResponse:
            ...  # on error, handler's return value IS the function's return value

    Side-effect-only handlers (CLI tools, task runners)::

        boundary = ErrorBoundary(exit_code=1)

        @boundary.handler(ValidationError)
        def handle_validation(exc: ValidationError) -> None:
            show_field_errors(exc)

        @boundary
        def main() -> None:
            ...

    Subprocess error handling (hooks, scripts)::

        boundary = ErrorBoundary(exit_code=2)

        @boundary
        def main() -> int:
            subprocess.run([...], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True)
            return 0

        @boundary.handler(subprocess.CalledProcessError)
        def _handle_subprocess(exc: subprocess.CalledProcessError) -> None:
            sys.stderr.buffer.write(exc.stdout)

    Use ``check=True`` and let ErrorBoundary handle the failure —
    don't manually check ``returncode`` inside ``@boundary`` functions.

    Per-exception exit codes (CLI tools with multiple outcomes)::

        boundary = ErrorBoundary(exit_code=1)  # default for unhandled errors

        class FixableIssue(Exception):
            pass

        class UnfixableIssue(Exception):
            pass

        @boundary.handler(FixableIssue)
        def handle_fixable(exc: FixableIssue) -> None:
            print(f'Fixable: {exc}', file=sys.stderr)
            sys.exit(10)  # overrides boundary's exit_code

        @boundary.handler(UnfixableIssue)
        def handle_unfixable(exc: UnfixableIssue) -> None:
            print(f'Unfixable: {exc}', file=sys.stderr)
            sys.exit(20)

        @boundary
        def main() -> None:
            ...  # FixableIssue -> exit 10, UnfixableIssue -> exit 20,
                 # unhandled Exception -> exit 1 (boundary default)

    This works because ``sys.exit()`` raises ``SystemExit`` (a BaseException),
    which propagates out of ``_handle`` before the boundary's default
    ``sys.exit(self._exit_code)`` executes.

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

    See ``scripts/claude-binary-patcher.py`` for the canonical real-world example.

Cross-language equivalents:
    React <ErrorBoundary> (renders fallback UI on error),
    Go defer+recover() (sets return value from deferred function),
    Rust panic::set_hook(), Elixir supervisor trees,
    Trio cancel scopes, Java UncaughtExceptionHandler.

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

type ErrorHandler = Callable[[Exception], Any]

_F = TypeVar('_F', bound=Callable[..., object])


class ErrorBoundary:
    """Error boundary with optional type-based handler dispatch.

    Catches application exceptions (Exception subclasses) and delegates to
    registered handlers. System exceptions pass through unconditionally.

    Two usage forms with different return semantics:

    - **Decorator**: ``@boundary`` — handler's return value becomes the
      function's return value. Use for HTTP handlers, task runners.
    - **Context manager**: ``with boundary:`` — exception is suppressed,
      handler's return value is discarded. Use for loop iterations.

    Type dispatch uses ``functools.singledispatch`` internally — handlers are
    matched by MRO, so registering for ``Exception`` acts as a catch-all.

    Args:
        handler: Convenience for registering a catch-all handler (equivalent
            to ``@boundary.handler(Exception)``). Defaults to printing
            the traceback to stderr.
        exit_code: Process exit code after handling. ``None`` for scope
            boundaries that should suppress and continue. Defaults to
            ``None`` (scope boundary).

    Examples:
        Scope boundary with fallback return value (decorator form)::

            boundary = ErrorBoundary()

            @boundary.handler(DatabaseError)
            def handle_db(exc: DatabaseError) -> Response:
                return ErrorResponse(str(exc))

            @boundary
            async def handle_request(request: Request) -> Response:
                ...  # if DatabaseError, returns the ErrorResponse

        Entry point with simple handler::

            @ErrorBoundary(exit_code=1, handler=log_error)
            def main() -> None:
                ...

        Scope boundary (context manager, suppress and continue)::

            for request in requests:
                with ErrorBoundary(handler=log_error):
                    process_request(request)

        Async entry point (auto-detected)::

            @ErrorBoundary(exit_code=1, handler=log_error)
            async def serve() -> None:
                await start_server()

    Composition:
        Error boundaries nest predictably. Inner boundaries handle first:

        - Inner with ``exit_code=1``: calls sys.exit(), outer sees SystemExit
          (BaseException, not Exception), passes through. Process exits.
        - Inner with ``exit_code=None``: suppresses, outer never sees it.
          Execution continues.
        - Handler calls ``sys.exit(N)``: same mechanism — SystemExit propagates
          out of ``_handle``, boundary's default exit code never fires.
    """

    def __init__(self, *, handler: ErrorHandler | None = None, exit_code: int | None = None) -> None:
        self._dispatch = singledispatch(_default_handler)
        if handler is not None:
            self._dispatch.register(Exception, handler)
        self._exit_code = exit_code

    def handler(
        self,
        exc_type: type[Exception],
    ) -> Callable[  # strict_typing_linter.py: loose-typing — generic decorator factory, return type must accept arbitrary callables
        [Callable[..., Any]],
        Callable[..., Any],
    ]:
        """Register a handler for a specific exception type.

        Uses ``functools.singledispatch`` for MRO-based matching — registering
        for ``Exception`` acts as a catch-all default.

        Handlers can return values. In decorator form, the return value becomes
        the decorated function's return value on error. In context manager form,
        the return value is discarded.

        Example::

            @boundary.handler(ValidationError)
            def handle_validation(exc: ValidationError) -> JSONResponse:
                return JSONResponse({'error': str(exc)})
        """
        return self._dispatch.register(exc_type)

    # -- Decorator protocol --

    def __call__(self, func: _F) -> _F:
        """Decorate a function with this error boundary.

        On error, the handler's return value becomes the function's return value.
        Auto-detects sync vs async and wraps accordingly. Parens required::

            @ErrorBoundary()          # correct
            @ErrorBoundary(handler=f) # correct
            @ErrorBoundary            # TypeError — parens required
        """
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:  # exception_safety_linter.py: swallowed-exception — ErrorBoundary delegates to handler which decides action
                    return self._handle(exc)

            return cast(_F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # exception_safety_linter.py: swallowed-exception — ErrorBoundary delegates to handler which decides action
                return self._handle(exc)

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
        if not isinstance(exc_value, Exception):
            return False
        self._handle(exc_value)
        return True

    # -- Async context manager --

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if not isinstance(exc_value, Exception):
            return False
        self._handle(exc_value)
        return True

    # -- Core logic --

    def _handle(self, exc: Exception) -> Any:
        """Dispatch to handler, optionally exit. Returns handler result.

        Handler failures cannot breach the boundary — if the registered handler
        raises an Exception, we fall back to stderr reporting of the original
        exception. If that also fails (e.g. stderr broken), we proceed silently
        with the configured suppress/exit behavior.

        Handlers CAN raise BaseException subclasses (SystemExit,
        KeyboardInterrupt) to bypass the boundary's default exit code.
        ``sys.exit(N)`` in a handler propagates past ``sys.exit(self._exit_code)``
        since the latter never executes. See "Per-exception exit codes" pattern.
        """
        result = None
        try:
            result = self._dispatch(exc)
        except Exception:  # exception_safety_linter.py: swallowed-exception — fallback to default handler
            try:  # noqa: SIM105 — intentional try/except/pass; contextlib.suppress hides the error-handler-of-last-resort intent
                _default_handler(exc)
            except Exception:  # exception_safety_linter.py: swallowed-exception — both handlers failed  # noqa: S110 — last resort error handler
                pass  # Both handlers failed (e.g. stderr broken); proceed with exit/suppress

        if self._exit_code is not None:
            sys.exit(self._exit_code)

        return result


def _default_handler(exc: Exception) -> None:
    """Print exception with traceback to stderr."""
    traceback.print_exception(exc, file=sys.stderr)
