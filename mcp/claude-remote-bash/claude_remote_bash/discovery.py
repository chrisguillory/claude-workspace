"""mDNS service registration and discovery via zeroconf."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping, Sequence

from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

__all__ = [
    'DiscoveredHost',
    'browse_hosts',
    'register_service',
    'resolve_host',
    'unregister_service',
]

SERVICE_TYPE = '_claude-rb._tcp.local.'  # DNS-SD: service type names <= 15 bytes
BROWSE_TIMEOUT_SECONDS = 3.0


class DiscoveredHost:
    """A daemon discovered via mDNS."""

    def __init__(self, *, alias: str, hostname: str, ip: str, port: int, os: str, user: str, version: str) -> None:
        self.alias = alias
        self.hostname = hostname
        self.ip = ip
        self.port = port
        self.os = os
        self.user = user
        self.version = version

    def __repr__(self) -> str:
        return f'DiscoveredHost(alias={self.alias!r}, ip={self.ip}, port={self.port})'


async def register_service(
    port: int,
    *,
    alias: str,
    version: str = '0.1.0',
) -> tuple[AsyncZeroconf, AsyncServiceInfo]:
    """Register this daemon as a mDNS service.

    Returns the AsyncZeroconf instance and ServiceInfo for later unregistration.
    The caller is responsible for calling ``unregister_service`` on shutdown.
    """
    hostname = socket.gethostname().removesuffix('.local')  # macOS includes .local already
    info = AsyncServiceInfo(
        type_=SERVICE_TYPE,
        name=f'{hostname}.{SERVICE_TYPE}',
        port=port,
        properties={
            'alias': alias,
            'os': 'darwin',
            'user': os.environ.get('USER', 'unknown'),
            'shell': os.environ.get('SHELL', '/bin/zsh'),
            'version': version,
        },
        server=f'{hostname}.local.',
        addresses=[socket.inet_aton(_local_ip())],
    )
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    await azc.async_register_service(info)
    return azc, info


async def unregister_service(azc: AsyncZeroconf, info: AsyncServiceInfo) -> None:
    """Unregister a previously registered service and close zeroconf."""
    await azc.async_unregister_service(info)
    await azc.async_close()


async def browse_hosts(timeout: float = BROWSE_TIMEOUT_SECONDS) -> Sequence[DiscoveredHost]:
    """Browse the LAN for claude-remote-bash daemons.

    Listens for mDNS advertisements for ``timeout`` seconds, resolves each
    discovered service, and returns a list of hosts with their metadata.
    """
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    hosts: list[DiscoveredHost] = []
    resolved_names: set[str] = set()
    pending: list[asyncio.Task[None]] = []

    async def _resolve(name: str) -> None:
        if name in resolved_names:
            return
        resolved_names.add(name)

        info = AsyncServiceInfo(SERVICE_TYPE, name)
        await info.async_request(azc.zeroconf, timeout=3000)

        addresses = info.parsed_addresses()
        if not addresses or info.port is None:
            return

        props = info.properties or {}
        hosts.append(
            DiscoveredHost(
                alias=_decode_prop(props, b'alias'),
                hostname=info.server or '',
                ip=addresses[0],
                port=info.port,
                os=_decode_prop(props, b'os'),
                user=_decode_prop(props, b'user'),
                version=_decode_prop(props, b'version'),
            )
        )

    def _on_change(
        *,
        zeroconf: object,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change is ServiceStateChange.Added:
            task = asyncio.ensure_future(_resolve(name))
            pending.append(task)

    browser = AsyncServiceBrowser(azc.zeroconf, SERVICE_TYPE, handlers=[_on_change])

    await asyncio.sleep(timeout)

    # Wait for any in-flight resolutions to complete
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    await browser.async_cancel()
    await azc.async_close()

    return hosts


def resolve_host(hosts: Sequence[DiscoveredHost], query: str) -> DiscoveredHost | None:
    """Resolve a host query against discovered hosts.

    Resolution chain:
        1. Exact alias match (case-insensitive)
        2. Substring match on hostname (case-insensitive)
        3. None if no match
    """
    query_lower = query.lower()

    for host in hosts:
        if host.alias.lower() == query_lower:
            return host

    for host in hosts:
        if query_lower in host.hostname.lower():
            return host

    return None


def _local_ip() -> str:
    """Discover the local LAN IP address via UDP connect to a public address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        host: str = s.getsockname()[0]
        return host
    finally:
        s.close()


def _decode_prop(props: Mapping[bytes, bytes | None], key: bytes) -> str:
    """Decode a TXT record property, defaulting to empty string."""
    val = props.get(key)
    if val is None:
        return ''
    return val.decode(errors='replace') if isinstance(val, bytes) else str(val)
