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

from claude_remote_bash.exceptions import ProtocolError

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

    legacy: bool
    """``True`` if the daemon advertised no ``if=`` TXT key. Interface kinds are
    not available; every address is assigned ``kind='other'``. Dispatch still
    works (it only needs IPs); audio orchestration refuses legacy hosts at the
    apply boundary since Ethernet preference requires real kind metadata."""


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

        addresses: list[DiscoveredAddress]
        legacy: bool
        if if_raw:
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
            legacy = False
        else:
            logger.warning(
                '%s: advertised no `if=` TXT key — legacy daemon, upgrade to advertise interface kinds',
                name,
            )
            addresses = [DiscoveredAddress(ip=ip, kind='other') for ip in ips]
            legacy = True

        addresses.sort(key=DiscoveredAddress.rank)

        found.append(
            DiscoveredHost(
                alias=_decode_prop(props, b'alias'),
                hostname=info.server or '',
                addresses=addresses,
                port=info.port,
                version=_decode_prop(props, b'version'),
                legacy=legacy,
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
    r'Hardware Port:\s*(?P<port>[^\n]+?)\s*\nDevice:\s*(?P<device>\S+)',
)

_zeroconf_log_filter_installed = False


def _partition(hosts: Sequence[DiscoveredHost]) -> BrowseResult:
    """Split discovered hosts into the local daemon (if any) and remote daemons.

    Match is by IP-string intersection against ``local_addresses()``. The first
    overlapping host becomes ``local_daemon``; any subsequent overlapping
    hosts (zombie mDNS records during daemon restart, etc.) fall through to
    ``remote_daemons`` rather than overwriting and silently disappearing.
    """
    local_ips = {a.ip for a in local_addresses()}
    local_daemon: DiscoveredHost | None = None
    remote: list[DiscoveredHost] = []
    for h in hosts:
        if local_daemon is None and local_ips.intersection(a.ip for a in h.addresses):
            local_daemon = h
        else:
            remote.append(h)
    return BrowseResult(remote_daemons=remote, local_daemon=local_daemon)


_TXT_VALUE_MAX_BYTES = 252
"""DNS TXT character-string limit (255) minus ``len('if=')`` — the budget for the encoded address list."""


def _encode_addresses_txt(addresses: Sequence[DiscoveredAddress]) -> str:
    """Serialize addresses into the TXT ``if=`` value: ``ip|kind,ip|kind,...``.

    Raises ``RuntimeError`` if the encoded value would exceed
    ``_TXT_VALUE_MAX_BYTES``. Without this guard, ``zeroconf`` raises a generic
    ``ValueError`` from inside its TXT packing routine that names the wrong
    layer; the daemon then fails to start with no actionable diagnostic.
    """
    value = ','.join(f'{a.ip}|{a.kind}' for a in addresses)
    if len(value.encode('utf-8')) > _TXT_VALUE_MAX_BYTES:
        raise ProtocolError(
            f'TXT `if=` value would be {len(value)} bytes for {len(addresses)} addresses; '
            f'DNS TXT character-strings cap at 255. Drop the lowest-rank addresses '
            f'(typically inactive VPN tunnels or stale Thunderbolt bridges) or stop '
            f'advertising over interfaces that this daemon does not need.'
        )
    return value


def _decode_addresses_txt(raw: str) -> Mapping[str, InterfaceKind]:
    """Parse a TXT ``if=`` value into ``{ip: kind}``.

    Wire format is daemon-to-daemon under our control. Malformed pairs and
    unknown kinds raise — they signal a bug in the advertising daemon's
    encoder or version drift that the operator needs to see, not absorb.
    """
    out: dict[str, InterfaceKind] = {}
    for pair_raw in raw.split(','):
        pair = pair_raw.strip()
        ip, sep, raw_kind = pair.partition('|')
        if not sep:
            raise ProtocolError(f'malformed TXT `if=` entry {pair!r}: missing `|` separator')
        if raw_kind not in _VALID_KINDS:
            raise ProtocolError(f'unknown InterfaceKind {raw_kind!r} in TXT `if=`; valid: {sorted(_VALID_KINDS)}')
        out[ip.strip()] = cast('InterfaceKind', raw_kind)
    return out


def _classify_adapters() -> Mapping[str, InterfaceKind]:
    """Build a ``{adapter_name: kind}`` map for every interface on this host.

    macOS: parses ``networksetup -listallhardwareports`` for hardware ports
    (Ethernet, Wi-Fi, Thunderbolt Bridge) and falls back to BSD-name-prefix
    matching for virtual tunnels (``utun*``, ``ipsec*``, ``wg*``, ``tailscale*``)
    that ``networksetup`` doesn't list.

    Non-macOS: returns an empty map; all interfaces classify as ``other`` until
    a platform-specific classifier lands. Errors from ``networksetup`` (missing
    binary, non-zero exit, timeout) propagate — a host that can't classify its
    own interfaces should not silently advertise them all as ``other``.
    """
    if sys.platform != 'darwin':
        return {}

    result = subprocess.run(
        ['networksetup', '-listallhardwareports'],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )

    out: dict[str, InterfaceKind] = dict(_parse_networksetup(result.stdout))
    for adapter in ifaddr.get_adapters():
        if adapter.name in out:
            continue
        out[adapter.name] = _kind_from_adapter_name(adapter.name)
    return out


def _kind_from_adapter_name(name: str) -> InterfaceKind:
    """Classify by BSD device name when no Hardware Port entry was found.

    macOS hides virtual tunnels (Tailscale, WireGuard, IPSec, corporate VPN)
    from ``networksetup``; their device names follow established conventions
    (``utun0``, ``wg0``, ``ipsec0``, ``tailscale0``) that we can match directly.
    """
    if name.startswith(('utun', 'ipsec', 'wg', 'tailscale')):
        return 'vpn'
    return 'other'


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
