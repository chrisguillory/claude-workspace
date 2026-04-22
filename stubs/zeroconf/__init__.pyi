"""Type stubs for zeroconf library (v0.148.0).

Partial coverage — only APIs used by claude-remote-bash.
"""

import enum
from typing import Any

class IPVersion(enum.Enum):
    All = ...
    V4Only = ...
    V6Only = ...

class ServiceStateChange(enum.Enum):
    Added = ...
    Removed = ...
    Updated = ...

class Zeroconf:
    def close(self) -> None: ...
    def get_service_info(self, type_: str, name: str, timeout: int = ...) -> ServiceInfo | None: ...

class ServiceInfo:
    name: str
    type: str
    port: int | None
    server: str | None
    properties: dict[bytes, bytes | None] | None

    def __init__(
        self,
        type_: str,
        name: str,
        port: int | None = ...,
        properties: dict[str, str] | None = ...,
        server: str | None = ...,
        addresses: list[bytes] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def parsed_addresses(self, version: IPVersion = ...) -> list[str]: ...
    def request(self, zc: Zeroconf, timeout: int = ...) -> bool: ...

class ServiceBrowser:
    def __init__(
        self,
        zc: Zeroconf,
        type_: str | list[str],
        handlers: list[Any] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def cancel(self) -> None: ...
