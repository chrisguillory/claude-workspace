from __future__ import annotations

import asyncio
import json
import unicodedata
from collections.abc import Sequence

from cc_lib.schemas import ClosedModel
from claude_remote_bash import DispatchService
from claude_remote_bash.dispatch import HostRunResult

__all__ = [
    'BluetoothDevice',
    'BluetoothError',
    'connect',
    'disconnect',
    'ensure_connected',
    'find_device',
    'is_connected',
    'list_devices',
    'steal',
    'where_is',
]


_DISPATCH_SESSION = 'claude-remote-audio-bluetooth'
_DISPATCH_TIMEOUT_S = 15.0


class BluetoothError(RuntimeError):
    """A Bluetooth operation failed in a way that needs the user to act."""


class BluetoothDevice(ClosedModel):
    """A paired Bluetooth audio device, observed on one specific host."""

    name: str
    """Display name from ``system_profiler`` — matches Core Audio's canonical name."""

    address: str
    """MAC address in blueutil's canonical form: lowercase, dash-separated."""

    minor_type: str
    """e.g. ``Headphones``, ``Speakers`` (from system_profiler's ``device_minorType``)."""

    services: str
    """Raw service string from system_profiler (e.g. ``HFP AVRCP A2DP``)."""

    connected: bool
    """Whether the device is currently BT-connected on the observed host."""


# -- Public API ---------------------------------------------------------------


async def list_devices(service: DispatchService, host_alias: str) -> Sequence[BluetoothDevice]:
    """List paired Bluetooth audio devices on ``host_alias``.

    Filters to devices that expose at least one audio profile (A2DP or HFP).
    Keyboards, mice, and other non-audio peripherals are excluded.
    """
    raw = (await _run(service, host_alias, 'system_profiler SPBluetoothDataType -json')).stdout
    return _parse_system_profiler(raw)


async def find_device(
    service: DispatchService,
    host_alias: str,
    name: str,
) -> BluetoothDevice | None:
    """Return the BT audio device with ``name`` on ``host_alias``, or ``None`` if not paired.

    Name matching uses the same NFKC + smart-quote fold as
    ``orchestrator._resolve_output_device`` — so a user-typed straight
    apostrophe matches Core Audio's curly form.
    """
    devices = await list_devices(service, host_alias)
    folded = _nfkc_casefold(name)
    return next((d for d in devices if _nfkc_casefold(d.name) == folded), None)


async def where_is(
    service: DispatchService,
    name: str,
    mesh_aliases: Sequence[str],
) -> Sequence[str]:
    """Return host aliases in ``mesh_aliases`` that currently have ``name`` BT-connected.

    Reads via ``system_profiler`` on each host in parallel — no TCC, no
    mutation. Individual host failures (offline, dispatch error) are
    swallowed so a partial result still reflects the reachable mesh.
    """
    results = await asyncio.gather(
        *(_is_connected_via_system_profiler(service, h, name) for h in mesh_aliases),
        return_exceptions=True,
    )
    return [h for h, r in zip(mesh_aliases, results, strict=True) if r is True]


async def connect(service: DispatchService, host_alias: str, address: str) -> None:
    """Engage the Bluetooth link to ``address`` on ``host_alias``.

    Brings up the BT control link (and A2DP if the device was previously
    registered with Core Audio on this host). Does NOT yield exclusive
    audio routing on its own — see ``steal``. Implementation shells out
    to ``blueutil``, so this requires macOS Bluetooth TCC granted.
    """
    await _run(service, host_alias, f'blueutil --connect {_normalize_address(address)}')


async def disconnect(service: DispatchService, host_alias: str, address: str) -> None:
    """Tear down the Bluetooth link to ``address`` on ``host_alias``."""
    await _run(service, host_alias, f'blueutil --disconnect {_normalize_address(address)}')


async def is_connected(service: DispatchService, host_alias: str, address: str) -> bool:
    """Return ``True`` if the Bluetooth link to ``address`` is currently up on ``host_alias``."""
    result = await _run(
        service,
        host_alias,
        f'blueutil --is-connected {_normalize_address(address)}',
    )
    return result.stdout.strip() == '1'


async def ensure_connected(
    service: DispatchService,
    host_alias: str,
    address: str,
) -> bool:
    """Connect device on host if not already connected. Idempotent.

    Returns ``True`` if a connect was performed, ``False`` if already
    connected. The caller is responsible for any post-connect wait — A2DP
    engagement can lag the BT-link by ~1-3s on a cold device.
    """
    if await is_connected(service, host_alias, address):
        return False
    await connect(service, host_alias, address)
    return True


async def steal(
    service: DispatchService,
    target_host: str,
    address: str,
    mesh_aliases: Sequence[str],
) -> Sequence[str]:
    """Move BT device to ``target_host`` with exclusive routing.

    Empirically validated algorithm:

    1. Find every host in the mesh currently BT-connected to the device.
    2. Disconnect on each of them (force-release).
    3. Connect on the target.

    Plain ``blueutil --connect`` on the target alone results in macOS
    Continuity multi-link sharing — audio mixes from all "connected" Macs.
    The disconnect-on-source step is what yields exclusive routing.

    Returns the list of host aliases the device was disconnected from
    (empty if it wasn't connected anywhere else).
    """
    sources = [h for h in mesh_aliases if h != target_host]
    if sources:
        owner_flags = await asyncio.gather(
            *(is_connected(service, h, address) for h in sources),
            return_exceptions=True,
        )
        owners = [h for h, r in zip(sources, owner_flags, strict=True) if r is True]
    else:
        owners = []

    if owners:
        await asyncio.gather(*(disconnect(service, h, address) for h in owners))

    await connect(service, target_host, address)
    return owners


# -- Internal helpers ---------------------------------------------------------


async def _run(
    service: DispatchService,
    host: str,
    command: str,
    timeout: float = _DISPATCH_TIMEOUT_S,
) -> HostRunResult:
    """Execute ``command`` on ``host``; raise ``BluetoothError`` on failure.

    Mirrors ``orchestrator._run``'s contract (raise on dispatch-level error or
    non-zero exit) but raises a domain-specific ``BluetoothError`` so callers
    can distinguish "the BT operation failed" from "the orchestrator's other
    work failed."
    """
    result = await service.run_target(
        host,
        command,
        session_id=_DISPATCH_SESSION,
        agent_id=None,
        timeout=timeout,
    )
    if not result.results:
        raise BluetoothError(f'{host}: no host results returned from dispatch')
    hr = result.results[0]
    if hr.error is not None:
        raise BluetoothError(f'{host}: dispatch failed: {hr.error}')
    if hr.exit_code != 0:
        raise BluetoothError(
            f'{host}: command failed (exit {hr.exit_code}): {command[:80]}\n  stderr: {hr.stderr.strip() or "<empty>"}'
        )
    return hr


def _parse_system_profiler(raw_json: str) -> Sequence[BluetoothDevice]:
    """Parse ``system_profiler SPBluetoothDataType -json`` into a ``BluetoothDevice`` list.

    The JSON shape is ``{SPBluetoothDataType: [controller]}``, where each
    controller has ``device_connected`` and ``device_not_connected`` arrays.
    Each array entry is a single-key dict whose key is the device name and
    whose value is a dict of device fields. Non-audio devices (no A2DP or HFP
    in their service list) are filtered out.
    """
    data = json.loads(raw_json)
    items = data.get('SPBluetoothDataType', [])
    if not items:
        return []
    controller = items[0]
    out: list[BluetoothDevice] = []
    for section_key, is_connected_section in (('device_connected', True), ('device_not_connected', False)):
        for entry in controller.get(section_key, []):
            for name, info in entry.items():
                services = info.get('device_services', '')
                if 'A2DP' not in services and 'HFP' not in services:
                    continue
                out.append(
                    BluetoothDevice(
                        name=name,
                        address=_normalize_address(info.get('device_address', '')),
                        minor_type=info.get('device_minorType', ''),
                        services=services,
                        connected=is_connected_section,
                    )
                )
    return out


async def _is_connected_via_system_profiler(
    service: DispatchService,
    host_alias: str,
    name: str,
) -> bool:
    """True if ``name`` appears in ``device_connected`` on ``host_alias`` (no TCC needed)."""
    devices = await list_devices(service, host_alias)
    folded = _nfkc_casefold(name)
    return any(d.connected and _nfkc_casefold(d.name) == folded for d in devices)


def _normalize_address(addr: str) -> str:
    """Normalize MAC address to blueutil's canonical form: lowercase, dash-separated."""
    return addr.replace(':', '-').lower()


_QUOTE_FOLD_TABLE = str.maketrans(
    {
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote / apostrophe (the Core Audio apostrophe)
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\u00a0': ' ',  # NBSP
    }
)


def _nfkc_casefold(s: str) -> str:
    """Same fold as ``orchestrator._nfkc_casefold``. Duplicated to avoid a circular import.

    If a third caller emerges, extract to a shared ``unicode_match`` module.
    """
    return unicodedata.normalize('NFKC', s).translate(_QUOTE_FOLD_TABLE).casefold()
