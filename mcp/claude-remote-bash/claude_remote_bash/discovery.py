"""mDNS service registration and discovery via zeroconf."""

from __future__ import annotations

import asyncio
import errno
import logging
import re
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Literal, cast, get_args

import ifaddr
from cc_lib.schemas import ClosedModel, StrictModel
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

__all__ = [
    'BrowseResult',
    'DiscoveredAddress',
    'DiscoveredHost',
    'InterfaceKind',
    'browse_hosts',
    'local_addresses',
    'register_service',
    'unregister_service',
]

logger = logging.getLogger(__name__)

SERVICE_TYPE = '_claude-rb._tcp.local.'  # DNS-SD: service type names <= 15 bytes
BROWSE_TIMEOUT_SECONDS = 3.0

type InterfaceKind = Literal['ethernet', 'wifi', 'thunderbolt', 'vpn', 'cellular', 'other']
"""Physical/logical kind of a network interface, normalized across OSes.

- ethernet:    any wired Layer 2 interface (built-in, USB-C adapter, USB-A adapter)
- wifi:        802.11 wireless
- thunderbolt: peer-to-peer Thunderbolt Bridge only
- vpn:         utun*, ipsec*, wireguard*, tailscale*
- cellular:    WWAN modems and tethered iPhones
- other:       fallback when the platform doesn't disclose enough to classify
"""


class DiscoveredAddress(StrictModel):
    """One IP a daemon advertises, paired with its interface kind."""

    ip: str
    """IPv4 address in dotted-decimal form."""

    kind: InterfaceKind
    """Physical/logical interface this address belongs to."""

    def rank(self) -> tuple[int, int, str]:
        """Sort key — interface kind beats subnet bucket beats lexicographic IP.

        Ethernet always sorts before Wi-Fi on the same host. Within a kind,
        common home-LAN subnets (192.168/16) sort before enterprise/VPN/CGNAT.
        """
        return (_KIND_RANK[self.kind], _subnet_bucket(self.ip), self.ip)


class DiscoveredHost(ClosedModel):
    """A daemon discovered via mDNS, with every address it advertises."""

    alias: str
    """User-facing short name configured per-daemon, e.g. ``M5``."""

    hostname: str
    """Fully-qualified ``.local.`` hostname the daemon advertises."""

    addresses: Sequence[DiscoveredAddress]
    """Every IP the daemon advertises, sorted with Ethernet first via ``DiscoveredAddress.rank``."""

    port: int
    """TCP port the daemon listens on for dispatch connections."""

    version: str
    """``claude-remote-bash`` package version of the advertising daemon."""


class BrowseResult(ClosedModel):
    """What ``browse_hosts`` learned about the LAN."""

    remote_daemons: Sequence[DiscoveredHost]
    """Daemons advertised by other machines on the LAN; never includes the local machine."""

    local_daemon: DiscoveredHost | None = None
    """The local machine's daemon if its mDNS advertisement is on the wire; ``None`` otherwise."""


async def browse_hosts(timeout: float = BROWSE_TIMEOUT_SECONDS) -> BrowseResult:
    """Browse the LAN for claude-remote-bash daemons and partition self from peers.

    Listens for mDNS advertisements for ``timeout`` seconds, resolves each
    discovered service, parses its TXT ``if=`` key into per-address interface
    kinds, and partitions the result. The local daemon (any IP overlap with
    ``local_addresses()``) is hoisted out of ``remote_daemons``.

    Daemons without a valid ``if=`` TXT key are skipped with a logged warning.
    """
    _install_zeroconf_log_filter()
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    found: list[DiscoveredHost] = []
    resolved_names: set[str] = set()
    pending: list[asyncio.Task[None]] = []

    async def _resolve(name: str) -> None:
        if name in resolved_names:
            return
        resolved_names.add(name)

        info = AsyncServiceInfo(SERVICE_TYPE, name)
        await info.async_request(azc.zeroconf, timeout=3000)

        ips = info.parsed_addresses()
        if not ips or info.port is None:
            return

        props = info.properties or {}
        if_raw = _decode_prop(props, b'if')
        if not if_raw:
            logger.warning(
                '%s: advertised no `if=` TXT key — upgrade this daemon to advertise interface kinds',
                name,
            )
            return

        kind_by_ip = _decode_addresses_txt(if_raw)
        addresses = []
        for ip in ips:
            kind = kind_by_ip.get(ip)
            if kind is None:
                logger.warning(
                    '%s: A record %s has no matching `if=` entry — daemon TXT is inconsistent, skipping',
                    name,
                    ip,
                )
                return
            addresses.append(DiscoveredAddress(ip=ip, kind=kind))

        addresses.sort(key=DiscoveredAddress.rank)

        found.append(
            DiscoveredHost(
                alias=_decode_prop(props, b'alias'),
                hostname=info.server or '',
                addresses=addresses,
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

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    await browser.async_cancel()
    await azc.async_close()

    return _partition(found)


def local_addresses() -> Sequence[DiscoveredAddress]:
    """Return every classifiable IPv4 address on this host, sorted by ``DiscoveredAddress.rank``.

    Excludes loopback (127.*) and link-local (169.254.*). Each address is paired
    with its interface kind via ``_classify_adapters``. Ethernet sorts first.
    """
    name_to_kind = _classify_adapters()
    out: list[DiscoveredAddress] = []
    for adapter in ifaddr.get_adapters():
        kind = name_to_kind.get(adapter.name, 'other')
        for addr in adapter.ips:
            if not addr.is_IPv4:
                continue
            ip = addr.ip
            if not isinstance(ip, str):
                continue
            if ip.startswith(('127.', '169.254.')):
                continue
            out.append(DiscoveredAddress(ip=ip, kind=kind))
    out.sort(key=DiscoveredAddress.rank)
    return out


async def register_service(
    port: int,
    *,
    alias: str,
    version: str,
) -> tuple[AsyncZeroconf, AsyncServiceInfo]:
    """Register this daemon as an mDNS service.

    Advertises every classifiable address the host has via A records, and a
    parallel ``if=ip|kind,ip|kind,...`` key in the TXT record so browsers can
    label each IP with its interface kind. The caller is responsible for
    calling ``unregister_service`` on shutdown.
    """
    _install_zeroconf_log_filter()
    hostname = socket.gethostname().removesuffix('.local')  # macOS includes .local already
    addrs = local_addresses()
    info = AsyncServiceInfo(
        type_=SERVICE_TYPE,
        name=f'{hostname}.{SERVICE_TYPE}',
        port=port,
        properties={
            'alias': alias,
            'version': version,
            'if': _encode_addresses_txt(addrs),
        },
        server=f'{hostname}.local.',
        addresses=[socket.inet_aton(a.ip) for a in addrs],
    )
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)
    await azc.async_register_service(info)
    return azc, info


async def unregister_service(azc: AsyncZeroconf, info: AsyncServiceInfo) -> None:
    """Unregister a previously registered service and close zeroconf."""
    await azc.async_unregister_service(info)
    await azc.async_close()


# -- Internal helpers ---------------------------------------------------------


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


_VALID_KINDS: frozenset[str] = frozenset(get_args(InterfaceKind.__value__))

_KIND_RANK: Mapping[InterfaceKind, int] = {
    'ethernet': 0,
    'thunderbolt': 1,
    'wifi': 2,
    'cellular': 3,
    'vpn': 4,
    'other': 5,
}

_NETWORKSETUP_BLOCK = re.compile(
    r'Hardware Port:\s*(?P<port>.+?)\s*\n.*?Device:\s*(?P<device>\S+)',
    re.DOTALL,
)

_zeroconf_log_filter_installed = False


def _partition(hosts: Sequence[DiscoveredHost]) -> BrowseResult:
    """Split discovered hosts into the local daemon (if any) and remote daemons.

    Match is by IP-string intersection against ``local_addresses()``; any
    single overlap identifies the host as the local daemon.
    """
    local_ips = {a.ip for a in local_addresses()}
    local_daemon: DiscoveredHost | None = None
    remote: list[DiscoveredHost] = []
    for h in hosts:
        if local_ips.intersection(a.ip for a in h.addresses):
            local_daemon = h
        else:
            remote.append(h)
    return BrowseResult(remote_daemons=remote, local_daemon=local_daemon)


def _encode_addresses_txt(addresses: Sequence[DiscoveredAddress]) -> str:
    """Serialize addresses into the TXT ``if=`` value: ``ip|kind,ip|kind,...``."""
    return ','.join(f'{a.ip}|{a.kind}' for a in addresses)


def _decode_addresses_txt(raw: str) -> Mapping[str, InterfaceKind]:
    """Parse a TXT ``if=`` value into ``{ip: kind}``.

    Unknown kinds are logged at WARNING and coerced to ``other`` so the host's
    other addresses remain usable. Silent fold without the warning would hide
    real cluster-wide drift.
    """
    out: dict[str, InterfaceKind] = {}
    for pair_raw in raw.split(','):
        pair = pair_raw.strip()
        ip, sep, raw_kind = pair.partition('|')
        if not sep:
            continue
        if raw_kind in _VALID_KINDS:
            out[ip.strip()] = cast('InterfaceKind', raw_kind)
        else:
            logger.warning('unknown InterfaceKind %r in TXT if=; classifying as other', raw_kind)
            out[ip.strip()] = 'other'
    return out


def _classify_adapters() -> Mapping[str, InterfaceKind]:
    """Build a ``{adapter_name: kind}`` map for every interface on this host.

    macOS: parses ``networksetup -listallhardwareports`` to map each BSD device
    name to its user-facing port name, then normalizes via ``_kind_from_port_name``.
    Substring-matched because macOS releases occasionally rename port strings
    and vendor adapters carry vendor names.

    Non-macOS: returns an empty map; all interfaces classify as ``other`` until
    a platform-specific classifier lands.
    """
    if sys.platform != 'darwin':
        return {}

    try:
        result = subprocess.run(
            ['networksetup', '-listallhardwareports'],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {}

    return _parse_networksetup(result.stdout)


def _parse_networksetup(text: str) -> Mapping[str, InterfaceKind]:
    """Parse ``networksetup -listallhardwareports`` stdout into ``{device: kind}``."""
    out: dict[str, InterfaceKind] = {}
    for match in _NETWORKSETUP_BLOCK.finditer(text):
        port_name = match.group('port').strip()
        device = match.group('device').strip()
        out[device] = _kind_from_port_name(port_name)
    return out


def _kind_from_port_name(name: str) -> InterfaceKind:
    """Normalize a macOS Hardware Port label into ``InterfaceKind``.

    Heuristics are substring-matched so vendor-specific USB-Ethernet adapter
    names (``USB 10/100/1G/2.5G LAN``, ``Apple USB Ethernet Adapter``, etc.)
    all classify as ``ethernet``. First hit wins. Thunderbolt-attached Ethernet
    adapters classify as ``ethernet``, not ``thunderbolt`` — only the
    peer-to-peer Thunderbolt Bridge port matches the ``thunderbolt`` arm.
    """
    lower = name.lower()
    if 'wi-fi' in lower or 'wifi' in lower or 'airport' in lower:
        return 'wifi'
    if 'thunderbolt bridge' in lower:
        return 'thunderbolt'
    if 'ethernet' in lower or 'lan' in lower:
        return 'ethernet'
    if 'iphone' in lower or 'cellular' in lower or 'modem' in lower or 'wwan' in lower:
        return 'cellular'
    return 'other'


def _subnet_bucket(ip: str) -> int:
    """Sub-rank for IPs within the same kind: home-LAN < enterprise < VPN < CGNAT < other."""
    if ip.startswith('192.168.'):
        return 0
    parts = ip.split('.', 2)
    if len(parts) < 2:
        return 4
    first = int(parts[0])
    second = int(parts[1])
    if first == 172 and 16 <= second <= 31:
        return 1
    if first == 10:
        return 2
    if first == 100 and 64 <= second <= 127:
        return 3
    return 4


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
