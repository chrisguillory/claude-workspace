"""Type stubs for circuitbreaker library (v2.1.3)."""

from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from datetime import datetime
from typing import Any, TypeVar

# Constants
STRING_TYPES: tuple[type[bytes], type[str]]
STATE_CLOSED: str
STATE_OPEN: str
STATE_HALF_OPEN: str

_F = TypeVar('_F', bound=Callable[..., Any])

def in_exception_list(
    *exc_types: type[Exception],
) -> Callable[[type[Exception], Exception], bool]: ...
def build_failure_predicate(
    expected_exception: type[Exception] | Iterable[type[Exception]] | Callable[[type[Exception], Exception], bool],
) -> Callable[[type[Exception], Exception], bool]: ...

class CircuitBreaker:
    # Class attributes (defaults)
    FAILURE_THRESHOLD: int
    RECOVERY_TIMEOUT: int
    EXPECTED_EXCEPTION: type[Exception]
    FALLBACK_FUNCTION: Callable[..., Any] | None

    # Instance attribute set in __init__
    is_failure: Callable[[type[Exception], Exception], bool]

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: int | None = None,
        expected_exception: type[Exception]
        | Iterable[type[Exception]]
        | Callable[[type[Exception], Exception], bool]
        | None = None,
        name: str | None = None,
        fallback_function: Callable[..., Any] | None = None,
    ) -> None: ...
    def __call__(self, wrapped: _F) -> _F: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _traceback: Any,
    ) -> bool: ...
    def __str__(self) -> str: ...
    def decorate(self, function: _F) -> _F: ...
    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    def call_generator(
        self, func: Callable[..., Generator[Any, Any, Any]], *args: Any, **kwargs: Any
    ) -> Generator[Any, Any, Any]: ...
    async def call_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...
    async def call_async_generator(
        self, func: Callable[..., AsyncGenerator[Any, Any]], *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, Any]: ...
    def reset(self) -> None: ...
    @property
    def state(self) -> str: ...
    @property
    def open_until(self) -> datetime: ...
    @property
    def open_remaining(self) -> int: ...
    @property
    def failure_count(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    @property
    def opened(self) -> bool: ...
    @property
    def name(self) -> str | None: ...
    @property
    def last_failure(self) -> BaseException | None: ...
    @property
    def fallback_function(self) -> Callable[..., Any] | None: ...

class CircuitBreakerError(Exception):
    _circuit_breaker: CircuitBreaker
    def __init__(self, circuit_breaker: CircuitBreaker, *args: Any, **kwargs: Any) -> None: ...
    def __str__(self) -> str: ...

class CircuitBreakerMonitor:
    circuit_breakers: dict[str | None, CircuitBreaker]
    @classmethod
    def register(cls, circuit_breaker: CircuitBreaker) -> None: ...
    @classmethod
    def all_closed(cls) -> bool: ...
    @classmethod
    def get_circuits(cls) -> Iterable[CircuitBreaker]: ...
    @classmethod
    def get(cls, name: str | bytes) -> CircuitBreaker | None: ...
    @classmethod
    def get_open(cls) -> Generator[CircuitBreaker]: ...
    @classmethod
    def get_closed(cls) -> Generator[CircuitBreaker]: ...

def circuit(
    failure_threshold: int | Callable[..., Any] | None = None,
    recovery_timeout: int | None = None,
    expected_exception: type[Exception]
    | Iterable[type[Exception]]
    | Callable[[type[Exception], Exception], bool]
    | None = None,
    name: str | None = None,
    fallback_function: Callable[..., Any] | None = None,
    cls: type[CircuitBreaker] = ...,
) -> CircuitBreaker: ...
