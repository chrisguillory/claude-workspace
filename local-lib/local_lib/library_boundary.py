"""Exception translation at third-party library call boundaries.

When calling third-party libraries, exceptions may escape outside the library's
own exception hierarchy. For example, bashlex raises ``NotImplementedError`` for
unimplemented features — a Python builtin indistinguishable from your own code's
``NotImplementedError``. LibraryBoundary translates any exception from a library
call into a known type, preserving the original via exception chaining.

This is distinct from ErrorBoundary (``error_boundary.py``), which catches and
handles exceptions at architectural edges. LibraryBoundary translates at call
sites — the translated exception still propagates to be handled by business
logic or ErrorBoundary.

Four layers of error handling:

    Layer 1 — Business Logic:
        Local try/except for expected domain errors.

    Layer 2 — Library Boundary:
        Translate third-party exceptions into domain types (this module).
        ``with boundary: lib.parse(cmd)``

    Layer 3 — Scope Boundary:
        ErrorBoundary(exit_code=None) at request/task edges.

    Layer 4 — Process Boundary:
        ErrorBoundary() at entry points.

Two modes:

    Explicit (context manager / decorator)::

        boundary = LibraryBoundary(BashlexError)

        with boundary:
            parts = bashlex.parse(command)

        @boundary
        async def fetch(url: str) -> Response:
            return await client.get(url)

    Automatic (proxy — wrap once, reassign, forget)::

        import bashlex

        bashlex = boundary.wrap(bashlex)
        parts = bashlex.parse(command)  # translated automatically

    The proxy wraps callables, generators, async generators, and context
    managers returned by library calls. Non-callable attributes pass through.

Production equivalents:
    Django DatabaseErrorWrapper, SQLAlchemy StatementError wrapping.

Anti-pattern to avoid:
    requests' inline cascading try/except (issue #1572) — enumerated catch
    clauses rot as the wrapped library evolves.

Cross-language equivalents:
    Rust From trait + ? operator, Go fmt.Errorf("%w") + errors.Is/As.

See also:
    - ``error_boundary.py``: Catches at architectural edges (Layers 3-4).
    - PEP 3134: Exception chaining (``raise X from Y``).
"""

from __future__ import annotations

__all__ = ['LibraryBoundary']

import asyncio
import functools
import inspect
import types
from collections.abc import Callable
from types import TracebackType
from typing import Any, Self, TypeVar, cast

_F = TypeVar('_F', bound=Callable[..., object])

# Control-flow exceptions that are Exception subclasses but must never be
# translated. StopIteration terminates iteration; StopAsyncIteration terminates
# async iteration; GeneratorExit closes generators. Translating any of these
# breaks Python's iterator/generator protocols.
_PASSTHROUGH = (StopIteration, StopAsyncIteration, GeneratorExit)


class LibraryBoundary:
    """Translate exceptions from third-party library calls into a known type.

    Catches any ``Exception`` that isn't already the target type and re-raises
    as the target with exception chaining (``raise X from Y``). System exceptions
    (``KeyboardInterrupt``, ``SystemExit``) and control-flow exceptions
    (``StopIteration``, ``GeneratorExit``) always pass through.

    Supports four usage forms:

    - **Context manager**: ``with boundary:`` or ``async with boundary:``
    - **Decorator**: ``@boundary`` on sync or async functions (auto-detected)
    - **Proxy**: ``boundary.wrap(library)`` for automatic wrapping

    Args:
        target: Exception type to translate into. Must accept a string message.

    Examples:
        Context manager (wrapping specific calls)::

            bashlex_call = LibraryBoundary(BashlexError)

            with bashlex_call:
                parts = bashlex.parse(command)

        Decorator (wrapping entire functions)::

            @LibraryBoundary(BashlexError)
            def parse_command(cmd: str) -> list:
                return bashlex.parse(cmd)

        Proxy (wrap once, reassign, forget)::

            import bashlex

            bashlex = LibraryBoundary(BashlexError).wrap(bashlex)
            parts = bashlex.parse(command)  # translated automatically
    """

    def __init__(self, target: type[Exception]) -> None:
        self._target = target

    # -- Decorator protocol --

    def __call__(self, func: _F) -> _F:
        """Decorate a function with this library boundary.

        Auto-detects sync vs async and wraps accordingly.
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
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_val is None or isinstance(exc_val, self._target):
            return  # No exception, or already translated (double-wrap guard)
        if not isinstance(exc_val, Exception) or isinstance(exc_val, _PASSTHROUGH):
            return  # System or control-flow exception — pass through
        original_message = str(exc_val)  # type preserved in __cause__, not here
        raise self._target(original_message).with_traceback(exc_tb) from exc_val

    # -- Async context manager --

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    # -- Proxy --

    def wrap(self, library: object) -> Any:
        """Wrap a library object so all calls translate exceptions automatically.

        Returns a proxy that intercepts attribute access. Callable attributes
        are wrapped to translate exceptions. Non-callable attributes pass through.

        The proxy also wraps generators, async generators, and context managers
        returned by library calls, so exceptions during iteration or context
        management are translated too.

        Limitation:
            Type checkers see ``Any`` for proxy attribute access.
        """
        return _TranslatingProxy(library, self)


# ---------------------------------------------------------------------------
# Proxy implementation
# ---------------------------------------------------------------------------


class _TranslatingProxy:
    """Proxy that wraps a library object, translating exceptions through a boundary.

    Detection strategy in ``__getattr__`` (function-level, most specific first):

    1. Not callable → return as-is
    2. Async generator function → plain wrapper, return ``_WrappedAsyncGenerator``
    3. Coroutine function → async wrapper, ``_wrap_result()``
    4. Generator function → plain wrapper, return ``_WrappedGenerator``
    5. Regular callable → sync wrapper, ``_wrap_result()``
    """

    def __init__(self, target: object, boundary: LibraryBoundary) -> None:
        object.__setattr__(self, '_target', target)
        object.__setattr__(self, '_boundary', boundary)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, '_target')
        boundary = object.__getattribute__(self, '_boundary')
        attr = getattr(target, name)

        if not callable(attr):
            return attr

        # Async generator function — wrapper is plain def (calling an async gen
        # returns the generator directly, no await needed).
        if inspect.isasyncgenfunction(attr):

            @functools.wraps(attr)
            def async_gen_wrapper(*args: Any, **kwargs: Any) -> _WrappedAsyncGenerator:
                return _WrappedAsyncGenerator(attr(*args, **kwargs), boundary)

            return async_gen_wrapper

        if asyncio.iscoroutinefunction(attr):

            @functools.wraps(attr)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with boundary:
                    result = await attr(*args, **kwargs)
                return self._wrap_result(result, boundary)

            return async_wrapper

        if inspect.isgeneratorfunction(attr):

            @functools.wraps(attr)
            def gen_wrapper(*args: Any, **kwargs: Any) -> _WrappedGenerator:
                with boundary:
                    gen = attr(*args, **kwargs)
                return _WrappedGenerator(gen, boundary)

            return gen_wrapper

        @functools.wraps(attr)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with boundary:
                result = attr(*args, **kwargs)
            return self._wrap_result(result, boundary)

        return sync_wrapper

    @staticmethod
    def _wrap_result(result: Any, boundary: LibraryBoundary) -> Any:
        """Wrap generators, async generators, and context managers returned by calls.

        Detection priority (most specific first):

        1. Generator — ``isinstance`` on ``GeneratorType`` (unambiguous)
        2. Async generator — ``isinstance`` on ``AsyncGeneratorType`` (unambiguous)
        3. Async context manager — has ``__aenter__``/``__aexit__``
        4. Sync context manager — has ``__enter__``/``__exit__``

        Generators are checked first because they are also context managers in
        some cases — the generator check is more specific.
        """
        if isinstance(result, types.GeneratorType):
            return _WrappedGenerator(result, boundary)
        if isinstance(result, types.AsyncGeneratorType):
            return _WrappedAsyncGenerator(result, boundary)
        if hasattr(result, '__aenter__') and hasattr(result, '__aexit__'):
            return _WrappedAsyncContextManager(result, boundary)
        if hasattr(result, '__enter__') and hasattr(result, '__exit__'):
            return _WrappedContextManager(result, boundary)
        return result


# ---------------------------------------------------------------------------
# Wrapper types — translate exceptions during iteration / context management
# ---------------------------------------------------------------------------


class _WrappedGenerator:
    """Wraps a sync generator, translating exceptions on every yield boundary."""

    __slots__ = ('_gen', '_boundary')

    def __init__(self, gen: types.GeneratorType[Any, Any, Any], boundary: LibraryBoundary) -> None:
        self._gen = gen
        self._boundary = boundary

    def __iter__(self) -> _WrappedGenerator:
        return self

    def __next__(self) -> Any:
        with self._boundary:
            return next(self._gen)

    def send(self, value: Any) -> Any:
        with self._boundary:
            return self._gen.send(value)

    def throw(self, value: BaseException) -> Any:
        """Forward throw to the underlying generator.

        If the generator raises a different exception in response, that IS a
        library exception and gets translated through the boundary.
        """
        with self._boundary:
            return self._gen.throw(value)

    def close(self) -> None:
        self._gen.close()


class _WrappedAsyncGenerator:
    """Wraps an async generator, translating exceptions on every yield boundary."""

    __slots__ = ('_gen', '_boundary')

    def __init__(self, gen: types.AsyncGeneratorType[Any, Any], boundary: LibraryBoundary) -> None:
        self._gen = gen
        self._boundary = boundary

    def __aiter__(self) -> _WrappedAsyncGenerator:
        return self

    async def __anext__(self) -> Any:
        async with self._boundary:
            return await self._gen.__anext__()

    async def asend(self, value: Any) -> Any:
        async with self._boundary:
            return await self._gen.asend(value)

    async def athrow(self, value: BaseException) -> Any:
        async with self._boundary:
            return await self._gen.athrow(value)

    async def aclose(self) -> None:
        await self._gen.aclose()


class _WrappedContextManager:
    """Wraps a sync context manager, translating exceptions in __enter__/__exit__."""

    __slots__ = ('_cm', '_boundary')

    def __init__(self, cm: Any, boundary: LibraryBoundary) -> None:
        self._cm = cm
        self._boundary = boundary

    def __enter__(self) -> Any:
        with self._boundary:
            return self._cm.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        with self._boundary:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)


class _WrappedAsyncContextManager:
    """Wraps an async context manager, translating exceptions in __aenter__/__aexit__."""

    __slots__ = ('_cm', '_boundary')

    def __init__(self, cm: Any, boundary: LibraryBoundary) -> None:
        self._cm = cm
        self._boundary = boundary

    async def __aenter__(self) -> Any:
        async with self._boundary:
            return await self._cm.__aenter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        async with self._boundary:
            return await self._cm.__aexit__(exc_type, exc_val, exc_tb)
