"""Type stubs for tenacity retry library."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar('_F', bound=Callable[..., Any])

def retry(
    *,
    wait: Any = None,
    stop: Any = None,
    reraise: bool = False,
    before_sleep: Callable[[RetryCallState], None] | None = None,
    **kwargs: Any,
) -> Callable[[_F], _F]: ...
def wait_exponential(
    multiplier: float = 1,
    min: float = 0,
    max: float = ...,
    exp_base: float = 2,
) -> Any: ...
def stop_after_attempt(max_attempt_number: int) -> Any: ...

class RetryCallState:
    attempt_number: int
    outcome: Any | None
    next_action: Any | None
    start_time: float
    retry_object: Any
    fn: Callable[..., Any] | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
