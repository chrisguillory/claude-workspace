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
    **kwargs: Any,
) -> Callable[[_F], _F]: ...
def wait_exponential(
    multiplier: float = 1,
    min: float = 0,
    max: float = ...,
    exp_base: float = 2,
) -> Any: ...
def stop_after_attempt(max_attempt_number: int) -> Any: ...
