"""A fake third-party library for testing LibraryBoundary.

Designed to look like a real library: module-level functions, module-level
attributes, factory functions returning context managers, generators, etc.
This is how bashlex, requests, sqlite3, etc. actually look.

Usage in tests::

    from tests.local_lib import fake_lib

    proxy = boundary.wrap(fake_lib)
    proxy.parse("hello")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

# -- Module-level attributes --

version = '1.0'
MAX_RETRIES = 3


# -- Sync functions --


def greet(name: str) -> str:
    """Return a greeting."""
    return f'Hello, {name}'


def fail() -> None:
    """Always raises ValueError."""
    raise ValueError('sync failure')


def fail_with_target(target_type: type[Exception]) -> None:
    """Raises the given exception type — for double-wrap guard tests."""
    raise target_type('already target')


def add(a: int, b: int) -> int:
    return a + b


# -- Sync generators --


def items() -> Iterator[int]:
    """Yields items then raises mid-iteration."""
    yield 1
    yield 2
    raise ValueError('mid-iteration failure')


def good_items() -> Iterator[int]:
    yield 10
    yield 20
    yield 30


def echo_gen() -> Iterator[str]:
    """Generator that echoes sent values."""
    value = yield 'ready'
    while True:
        value = yield f'echo:{value}'


def throw_converter() -> Iterator[str]:
    """Generator that converts thrown exceptions into ValueError."""
    while True:
        try:
            yield 'waiting'
        except RuntimeError:
            raise ValueError('converted from RuntimeError')


def fail_during_creation() -> Iterator[None]:
    """Generator function that raises BEFORE yielding anything."""
    raise ValueError('creation failure')
    yield  # pragma: no cover — makes this a generator function


# -- Async functions --


async def async_greet(name: str) -> str:
    return f'Hello async, {name}'


async def async_fail() -> None:
    raise ValueError('async failure')


# -- Async generators --


async def async_items() -> AsyncIterator[int]:
    yield 1
    yield 2
    raise ValueError('async mid-iteration failure')


async def async_good_items() -> AsyncIterator[int]:
    yield 10
    yield 20


# -- Context managers returned by factory functions --


class _Connection:
    """Simulates a database/network connection context manager."""

    def __enter__(self) -> str:
        return 'connected'

    def __exit__(self, *args: object) -> None:
        pass


class _FailingConnection:
    """Connection that fails on enter."""

    def __enter__(self) -> str:
        raise ConnectionError('connect failed')

    def __exit__(self, *args: object) -> None:
        pass


class _ExitFailingConnection:
    """Connection that fails on exit (cleanup error)."""

    def __enter__(self) -> str:
        return 'connected'

    def __exit__(self, *args: object) -> bool:
        raise ConnectionError('disconnect failed')


class _SuppressingConnection:
    """Connection whose __exit__ returns True, suppressing exceptions."""

    def __enter__(self) -> str:
        return 'suppressed'

    def __exit__(self, *args: object) -> bool:
        return True


class _ExitRaisesOnBodyError:
    """Connection whose __exit__ raises a NEW exception when body fails.

    Simulates a library that raises its own error during cleanup when the
    body has already failed (e.g. rollback failure after query failure).
    """

    def __enter__(self) -> str:
        return 'resource'

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_val is not None:
            raise ConnectionError('cleanup failed during error handling')


class _AsyncConnection:
    """Async connection that fails on enter."""

    async def __aenter__(self) -> str:
        raise ConnectionError('async connect failed')

    async def __aexit__(self, *args: object) -> None:
        pass


class _GoodAsyncConnection:
    async def __aenter__(self) -> str:
        return 'async connected'

    async def __aexit__(self, *args: object) -> None:
        pass


# -- Factory functions (module-level, like sqlite3.connect()) --


def connect() -> _Connection:
    return _Connection()


def connect_failing() -> _FailingConnection:
    return _FailingConnection()


def connect_exit_failing() -> _ExitFailingConnection:
    return _ExitFailingConnection()


def connect_suppressing() -> _SuppressingConnection:
    return _SuppressingConnection()


def connect_exit_raises_on_body_error() -> _ExitRaisesOnBodyError:
    return _ExitRaisesOnBodyError()


def async_connect_failing() -> _AsyncConnection:
    return _AsyncConnection()


def async_connect() -> _GoodAsyncConnection:
    return _GoodAsyncConnection()


# -- Callable objects (class instances with __call__) --


class Processor:
    """Callable object — has __call__ but is not a function."""

    def __call__(self, data: str) -> str:
        return f'processed:{data}'


class FailingProcessor:
    """Callable object that raises on call."""

    def __call__(self, data: str) -> str:
        raise ValueError(f'processing failed: {data}')


processor = Processor()
failing_processor = FailingProcessor()


# -- Objects with properties --


class Config:
    """Object with @property and regular attributes."""

    name = 'default'

    @property
    def computed(self) -> str:
        return f'computed:{self.name}'

    @property
    def failing_property(self) -> str:
        raise ValueError('property access failed')

    def method(self) -> str:
        return 'method result'


config = Config()


# -- Objects with magic methods --


class DataStore:
    """Object with __len__, __getitem__, __contains__, __iter__."""

    def __init__(self) -> None:
        self._data = [10, 20, 30]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> int:
        return self._data[index]

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def query(self) -> list[int]:
        return list(self._data)


data_store = DataStore()


# -- Submodule simulation --


class _SubModule:
    """Simulates a submodule (like bashlex.errors)."""

    error_code = 42

    @staticmethod
    def parse(text: str) -> str:
        return f'parsed:{text}'

    @staticmethod
    def fail() -> None:
        raise ValueError('submodule failure')


sub = _SubModule()
