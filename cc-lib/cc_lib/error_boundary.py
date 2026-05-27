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
            '''On error, the handler's return value IS this function's return value.'''

    Side-effect-only handlers (CLI tools, task runners)::

        boundary = ErrorBoundary(exit_code=1)


        @boundary.handler(ValidationError)
        def handle_validation(exc: ValidationError) -> None:
            show_field_errors(exc)


        @boundary
        def main() -> None:
            '''Real entry-point logic here.'''

    Subprocess error handling (hooks, scripts)::

        boundary = ErrorBoundary(exit_code=2)


        @boundary
        def main() -> int:
            subprocess.run([...], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
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
            '''FixableIssue -> exit 10, UnfixableIssue -> exit 20,
            unhandled Exception -> exit 1 (boundary default).'''

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

See Also:
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
    'render_recovery',
]

import asyncio
import functools
import sys
import traceback
from collections.abc import Callable
from functools import singledispatch
from types import TracebackType
from typing import Any, Self, TextIO, TypeGuard, TypeVar, cast, get_args

from cc_lib.claude_context import in_claude_code
from cc_lib.exceptions import ResolvableError
from cc_lib.types import OutputFormat

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
                '''If DatabaseError, returns the ErrorResponse from the handler above.'''

        Entry point with simple handler::

            @ErrorBoundary(exit_code=1, handler=log_error)
            def main() -> None:
                '''Real entry-point logic here.'''

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

    def __init__(
        self,
        *,
        handler: ErrorHandler | None = None,
        exit_code: int | None = None,
        format_resolver: Callable[[], str | None] | None = None,
    ) -> None:
        self._dispatch = singledispatch(_default_handler)
        if handler is not None:
            self._dispatch.register(Exception, handler)
        self._exit_code = exit_code
        self._format_resolver = format_resolver
        # Per-format singledispatch tables for format-narrowed handlers. Each
        # uses ``_no_handler`` as the default so dispatch results can be
        # compared with ``is _no_handler`` to tell "registered" from "not
        # registered" without a separate membership check.
        self._format_dispatchers: dict[OutputFormat, Any] = {}

    def handler(
        self,
        exc_type: type[Exception],
        *,
        format: OutputFormat | None = None,  # noqa: A002 — name matches the user-facing CLI flag
    ) -> Callable[  # strict_typing_linter.py: loose-typing — generic decorator factory, return type must accept arbitrary callables
        [Callable[..., Any]],
        Callable[..., Any],
    ]:
        """Register a handler for a specific exception type, optionally narrowed by format.

        Two registration modes share one decorator:

        - ``handler(exc_type)`` — format-agnostic. Fires for any active format
          (and for boundaries without a ``format_resolver``). Use for hooks,
          scripts, and CLIs that don't expose ``--format``.
        - ``handler(exc_type, format='text')`` — format-narrowed. Fires only
          when the active format matches. Resolves the ``(exception_type,
          output_format)`` cross-cutting concern that pure type-dispatch
          can't reach alone.

        Precedence at handle time: format-narrowed wins over format-agnostic,
        which wins over ``_default_handler``. Within each layer, dispatch is
        ``functools.singledispatch`` (MRO-based) — registering for
        ``Exception`` acts as a catch-all for that layer.

        Architectural exemplar: FastAPI's ``async def handler(request, exc)``
        signature, where the request — bound at dispatch time — gives global
        handlers access to per-invocation context. ``format_resolver`` is the
        CLI analog: registration is global, format is per-invocation, lookup
        happens at handle time.

        Registration accepts ``OutputFormat`` (the workspace's
        ``Literal['text', 'json']`` alias) so mypy catches typos at the
        decoration site, and internal storage is typed the same. The
        resolver-side interface returns ``str`` because resolvers can't know
        which CLI's format vocabulary is active — a CLI with extra formats
        (e.g. lineage's ``Literal['text', 'tree', 'json']``) may legitimately
        return ``'tree'``. ``_handle`` narrows the resolver's ``str`` to
        ``OutputFormat`` via ``_is_output_format`` at lookup; values outside
        the workspace standard fall through to the format-agnostic layer
        rather than failing the boundary.

        Handlers can return values. In decorator form, the return value
        becomes the decorated function's return value on error. In context
        manager form, the return value is discarded.

        Example::

            # Format-agnostic — fires for any format
            @boundary.handler(ValidationError)
            def handle_validation(exc: ValidationError) -> JSONResponse:
                return JSONResponse({'error': str(exc)})


            # Format-narrowed — fires only when active format matches
            @boundary.handler(ApplyError, format='text')
            def _render_apply_text(exc: ApplyError) -> None:
                print(f'apply: {exc}', file=sys.stderr)


            @boundary.handler(Exception, format='json')
            def _render_json(exc: Exception) -> None:
                print(json.dumps({'error': str(exc)}))
        """
        if format is None:
            return self._dispatch.register(exc_type)
        if format not in self._format_dispatchers:
            self._format_dispatchers[format] = singledispatch(_no_handler)
        # cast because dict[OutputFormat, Any] erases the singledispatch's
        # static type; the inferred `.register` return matches the declared
        # decorator-factory signature, the dict storage just hides it.
        return cast(
            Callable[[Callable[..., Any]], Callable[..., Any]], self._format_dispatchers[format].register(exc_type)
        )

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
        """Dispatch to a format-narrowed or format-agnostic handler, optionally exit.

        Lookup precedence:

        1. Format-narrowed handler registered for ``(type(exc), active_format)``,
           if a ``format_resolver`` is configured and returned a recognized
           format. Wins over format-agnostic.
        2. Format-agnostic handler registered for ``type(exc)`` (the
           ``handler(exc_type)`` no-kwargs form). The right tool for hooks,
           scripts, and CLIs without ``--format``.
        3. ``_default_handler`` (traceback to stderr) as last resort.

        Handler failures cannot breach the boundary — if the chosen handler
        raises an Exception, we fall back to ``_default_handler``. If that
        also fails (e.g. stderr broken), we proceed silently with the
        configured suppress/exit behavior.

        Handlers CAN raise BaseException subclasses (SystemExit,
        KeyboardInterrupt) to bypass the boundary's default exit code.
        ``sys.exit(N)`` in a handler propagates past ``sys.exit(self._exit_code)``
        since the latter never executes. See "Per-exception exit codes" pattern.

        Handler return values flow through both paths — a format-narrowed
        handler that returns a ``Response`` is just as valid as a
        format-agnostic one. The ``_NO_FORMAT_MATCH`` sentinel distinguishes
        "no narrowed handler matched, try format-agnostic" from "narrowed
        handler ran and returned ``None``."
        """
        result = None
        try:
            result = self._dispatch_format_narrowed(exc)
            if result is _NO_FORMAT_MATCH:
                result = self._dispatch(exc)
        except Exception:  # exception_safety_linter.py: swallowed-exception — fallback to default handler
            try:  # noqa: SIM105 — intentional try/except/pass; contextlib.suppress hides the error-handler-of-last-resort intent
                _default_handler(exc)
            except Exception:  # exception_safety_linter.py: swallowed-exception — both handlers failed  # noqa: S110 — last resort error handler
                pass  # Both handlers failed (e.g. stderr broken); proceed with exit/suppress

        if self._exit_code is not None:
            sys.exit(self._exit_code)

        return result

    def _dispatch_format_narrowed(self, exc: Exception) -> Any:
        """Run a format-narrowed handler for ``(type(exc), active_format)`` if registered.

        Returns the handler's result, or ``_NO_FORMAT_MATCH`` sentinel when:
        - no ``format_resolver`` was configured, OR
        - the resolver raised or returned ``None``, OR
        - the returned format isn't in ``OutputFormat`` (e.g. lineage's
          ``'tree'`` falls through to format-agnostic text rendering), OR
        - the format has no entry in ``_format_dispatchers``, OR
        - the per-format dispatch returned the ``_no_handler`` default.

        Handler exceptions propagate to ``_handle``'s outer try/except so
        they fall back to ``_default_handler`` — same shape as format-agnostic
        handler failures. Resolver failures collapse to ``_NO_FORMAT_MATCH``
        so a busted resolver never breaches the boundary.
        """
        if self._format_resolver is None:
            return _NO_FORMAT_MATCH
        try:
            fmt = self._format_resolver()
        except (
            Exception
        ):  # exception_safety_linter.py: swallowed-exception — resolver failure degrades to format-agnostic dispatch
            return _NO_FORMAT_MATCH
        if fmt is None or not _is_output_format(fmt):
            return _NO_FORMAT_MATCH
        if fmt not in self._format_dispatchers:
            return _NO_FORMAT_MATCH
        cb = self._format_dispatchers[fmt].dispatch(type(exc))
        if cb is _no_handler:
            return _NO_FORMAT_MATCH
        return cb(exc)


def render_recovery(exc: ResolvableError, *, stream: TextIO | None = None) -> None:
    """Emit a context-aware recovery footer for a ``ResolvableError`` to ``stream``.

    Three rendering modes, decided by environment, in precedence order:

    1. Inside Claude Code (``CLAUDECODE=1``) — emit a parseable agent-engagement
       block. The block names the ``code``, lists ``suggestions`` numbered, and
       surfaces ``docs_url`` if present. Tells the in-loop LLM that this is a
       recognized failure pattern and how to engage.
    2. Bare terminal (stream is a TTY, not in Claude Code) — emit a red-text
       block with the same structured info plus a CTA pointing the human at
       Claude Code for adaptive diagnosis.
    3. Piped / non-TTY output — emit nothing. The exception's ``__str__`` (the
       upstream rendering) already carries the message; appending decoration to
       piped output is noise.

    The structured fields on ``exc`` are the source of truth — this function
    renders them; it doesn't add information. Consumers that need a different
    medium (JSON output, log records, MCP tool-result blocks) read the fields
    directly and render their own way.
    """
    target = stream if stream is not None else sys.stderr
    if in_claude_code():
        _write_claude_code_footer(exc, target)
    elif target.isatty():
        _write_tty_footer(exc, target)


def _write_claude_code_footer(exc: ResolvableError, stream: TextIO) -> None:
    """Emit the agent-engagement footer to ``stream``. No ANSI — Claude Code reads plain."""
    stream.write(f'\n[Claude Code: resolvable error (code={exc.code}).')
    if exc.title:
        stream.write(f' {exc.title}.')
    stream.write('\n')
    if exc.suggestions:
        stream.write('Suggested steps:\n')
        for i, step in enumerate(exc.suggestions, 1):
            stream.write(f'  {i}. {step}\n')
    if exc.docs_url:
        stream.write(f'Workflow: {exc.docs_url}\n')
    stream.write('Apply judgment: follow the steps; fetch the workflow if needed; ')
    stream.write('ask the user for context (recent state changes, env shifts) when ambiguous.]\n')


def _write_tty_footer(exc: ResolvableError, stream: TextIO) -> None:
    """Emit the human-readable red footer with structured info + escalation CTA."""
    red, reset = '\033[31m', '\033[0m'
    stream.write(f'\n{red}── resolvable error: code={exc.code}')
    if exc.title:
        stream.write(f' — {exc.title}')
    stream.write(f' ──{reset}\n')
    if exc.suggestions:
        for i, step in enumerate(exc.suggestions, 1):
            stream.write(f'  {i}. {step}\n')
    if exc.docs_url:
        stream.write(f'  Workflow: {exc.docs_url}\n')
    stream.write(
        f'{red}For adaptive diagnosis, run from a Claude Code session, or paste this output into one.{reset}\n'
    )


def _default_handler(exc: Exception) -> None:
    """Print exception with traceback to stderr."""
    traceback.print_exception(exc, file=sys.stderr)


def _no_handler(exc: Exception) -> None:
    """Sentinel default for per-format ``singledispatch`` tables.

    Identity-comparable from ``_dispatch_format_narrowed`` to distinguish
    "no handler registered for this exception type at this format" from a
    real registered handler that happens to return ``None``.
    """


# Sentinel returned by ``_dispatch_format_narrowed`` when no format-narrowed
# handler ran. Identity-comparable; never collides with legitimate handler
# return values (including ``None``, which a real handler might intentionally
# return).
_NO_FORMAT_MATCH = object()


def _is_output_format(value: str) -> TypeGuard[OutputFormat]:
    """Narrow ``str`` to ``OutputFormat`` at runtime — the workspace's text/json gate.

    ``ErrorBoundary._dispatch_format_narrowed`` calls this on the resolver's
    return so a CLI with extra formats (lineage's ``'tree'``, playwright's
    ``'html'``/``'markdown'``) passes through the boundary gracefully — values
    outside ``OutputFormat`` miss the gate and fall through to format-agnostic
    handlers rather than failing the boundary or forcing a cast.
    """
    return value in get_args(OutputFormat)
