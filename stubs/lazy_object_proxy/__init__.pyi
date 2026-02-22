"""Type stubs for lazy-object-proxy."""

from collections.abc import Callable
from typing import Any

class Proxy:
    def __init__(self, factory: Callable[[], Any]) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
