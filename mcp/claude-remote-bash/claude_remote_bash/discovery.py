"""mDNS service registration and discovery via zeroconf."""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
from collections.abc import Mapping, Sequence

import ifaddr
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

__all__ = [
    'DiscoveredHost',
    'browse_hosts',
    'publishable_ipv4s',
    'register_service',
    'resolve_host',
    'unregister_service',
]

SERVICE_TYPE = '_claude-rb._tcp.local.'  # DNS-SD: service type names <= 15 bytes
BROWSE_TIMEOUT_SECONDS = 3.0


class DiscoveredHost:
    """A daemon discovered via mDNS.

    A single host may advertise multiple IP addresses (LAN + VPN + ethernet);
    callers should try each in order until one accepts a TCP connection.
    """

    def __init__(self, *, alias: str, hostname: str, ips: Sequence[str], port: int, version: str) -> None:
        self.alias = alias
        self.hostname = hostname
        self.ips = list(ips)
        self.port = port
        self.version = version

    def __repr__(self) -> str:
        return f'DiscoveredHost(alias={self.alias!r}, ips={self.ips}, port={self.port})'


async def register_service(
    port: int,
    *,
    alias: str,
    version: str = '0.1.0',
) -> tuple[AsyncZeroconf, AsyncServiceInfo]:
    """Register this daemon as a mDNS service.

    Returns the AsyncZeroconf instance and ServiceInfo for later unregistration.
    The caller is responsible for calling ``unregister_service`` on shutdown.

    Advertises all non-loopback, non-link-local IPv4 addresses the host has.
    Clients iterate through them at connect time, so LAN/VPN/ethernet all
    appear in the mesh without operator intervention.
    """
    _install_zeroconf_log_filter()
    hostname = socket.gethostname().removesuffix('.local')  # macOS includes .local already
    info = AsyncServiceInfo(
        type_=SERVICE_TYPE,
        name=f'{hostname}.{SERVICE_TYPE}',
        port=port,
        properties={
            'alias': alias,
            'version': version,
        },
        server=f'{hostname}.local.',
        addresses=[socket.inet_aton(ip) for ip in publishable_ipv4s()],
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
    _install_zeroconf_log_filter()
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

        # Sort client-side: zeroconf does not preserve the address order the
        # daemon registered, so the connect-attempt preference (LAN before
        # VPN) has to be re-applied here.
        addresses = sorted(addresses, key=_ipv4_rank)

        props = info.properties or {}
        hosts.append(
            DiscoveredHost(
                alias=_decode_prop(props, b'alias'),
                hostname=info.server or '',
                ips=addresses,
                port=info.port,
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


def publishable_ipv4s() -> Sequence[str]:
    """Return all non-loopback, non-link-local IPv4 addresses for this host.

    Addresses are sorted to prefer common home-LAN ranges before VPN/tunnel
    ranges. The full list is published via mDNS so clients can try each in
    turn; the sort order just decides which IP the client attempts first.

    Preference order (lowest-numbered rank is tried first):
        0 - 192.168.0.0/16      (consumer LAN)
        1 - 172.16.0.0/12       (enterprise LAN / Docker)
        2 - 10.0.0.0/8          (VPN, corp LAN, Tailscale-adjacent)
        3 - 100.64.0.0/10       (CGNAT / Tailscale)
        4 - everything else     (public IPv4, etc.)
    """
    ips: set[str] = set()
    for adapter in ifaddr.get_adapters():
        for addr in adapter.ips:
            if not addr.is_IPv4:
                continue
            ip = addr.ip
            if not isinstance(ip, str):
                continue
            if ip.startswith(('127.', '169.254.')):
                continue
            ips.add(ip)
    return sorted(ips, key=_ipv4_rank)


class _QuietUnreachableInterfaces(logging.Filter):
    """Drop ENETUNREACH / EHOSTUNREACH / EADDRNOTAVAIL from zeroconf.

    Zeroconf iterates every network interface at startup and attempts to bind
    a multicast socket on each. On a machine with VPN tunnels or offline
    interfaces this produces a scary-looking warning with traceback per bad
    interface — but the registration succeeds on whatever interfaces DO work,
    and the failures are entirely expected. Suppressing them by errno (rather
    than message text) is robust to zeroconf version changes.
    """

    _QUIET_ERRNOS = frozenset({errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EADDRNOTAVAIL})

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.exc_info:
            return True
        exc = record.exc_info[1]
        if isinstance(exc, OSError) and exc.errno in self._QUIET_ERRNOS:
            return False
        return True


_zeroconf_log_filter_installed = False


def _ipv4_rank(ip: str) -> tuple[int, str]:
    """Sort key for IPv4 addresses — prefer home-LAN ranges over VPN/tunnel."""
    if ip.startswith('192.168.'):
        return (0, ip)
    first = int(ip.split('.', 1)[0])
    second = int(ip.split('.', 2)[1])
    if first == 172 and 16 <= second <= 31:
        return (1, ip)
    if first == 10:
        return (2, ip)
    if first == 100 and 64 <= second <= 127:
        return (3, ip)
    return (4, ip)


def _decode_prop(props: Mapping[bytes, bytes | None], key: bytes) -> str:
    """Decode a TXT record property, defaulting to empty string."""
    val = props.get(key)
    if val is None:
        return ''
    return val.decode(errors='replace') if isinstance(val, bytes) else str(val)


def _install_zeroconf_log_filter() -> None:
    """Install the unreachable-interface filter on the zeroconf logger once per process."""
    global _zeroconf_log_filter_installed  # noqa: PLW0603 — single-writer idempotency flag
    if _zeroconf_log_filter_installed:
        return
    logging.getLogger('zeroconf').addFilter(_QuietUnreachableInterfaces())
    _zeroconf_log_filter_installed = True
