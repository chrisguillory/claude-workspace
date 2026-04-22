"""Type stubs for zeroconf.asyncio (v0.148.0).

Partial coverage — only APIs used by claude-remote-bash.
"""

from typing import Any

from zeroconf import IPVersion, ServiceInfo, Zeroconf

class AsyncZeroconf:
    zeroconf: Zeroconf

    def __init__(self, *, ip_version: IPVersion = ..., **kwargs: Any) -> None: ...
    async def async_register_service(self, info: AsyncServiceInfo, **kwargs: Any) -> None: ...
    async def async_unregister_service(self, info: AsyncServiceInfo, **kwargs: Any) -> None: ...
    async def async_close(self) -> None: ...

class AsyncServiceInfo(ServiceInfo):
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
    async def async_request(self, zc: Zeroconf, timeout: int = ...) -> bool: ...

class AsyncServiceBrowser:
    def __init__(
        self,
        zc: Zeroconf,
        type_: str | list[str],
        handlers: list[Any] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    async def async_cancel(self) -> None: ...
