from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections.abc import Sequence

from cc_lib.schemas import ClosedModel
from cc_lib.utils.unicode_match import nfkc_casefold
from claude_remote_bash import DispatchService
from claude_remote_bash.dispatch import HostRunResult

from claude_remote_audio.exceptions import BluetoothError

__all__ = [
    'BluetoothDevice',
    'connect',
    'disconnect',
    'engage_via_sound_menu',
    'ensure_connected',
    'find_device',
    'is_connected',
    'list_devices',
    'steal',
    'where_is',
]


_DISPATCH_SESSION = 'claude-remote-audio-bluetooth'
_DISPATCH_TIMEOUT_S = 15.0

logger = logging.getLogger(__name__)


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
    folded = nfkc_casefold(name)
    return next((d for d in devices if nfkc_casefold(d.name) == folded), None)


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
        owners = []
        for h, r in zip(sources, owner_flags, strict=True):
            if r is True:
                owners.append(h)
            elif isinstance(r, BaseException):
                # `is_connected` probe failed (transient dispatch hiccup, host
                # unreachable). Steal's exclusivity contract requires complete
                # owner-set knowledge — silently treating a probe failure as
                # "not connected" risks leaving the host BT-linked and
                # multi-link audio mixing from Continuity. Log so the apply
                # log records the unknown without raising (a transient probe
                # failure shouldn't abort an otherwise-recoverable steal).
                logger.warning('%s: is_connected probe failed; possible owner not disconnected: %s', h, r)
    else:
        owners = []

    if owners:
        logger.info('%s: stealing %s from %s', target_host, address, owners)
        await asyncio.gather(*(disconnect(service, h, address) for h in owners))
    else:
        logger.info('%s: claiming %s (no other owners)', target_host, address)

    await connect(service, target_host, address)
    return owners


async def engage_via_sound_menu(
    service: DispatchService,
    host_alias: str,
    name: str,
) -> bool:
    """Promote a Bluetooth audio device into Core Audio by simulating a Sound-menu click.

    macOS has two views of audio devices: Core Audio's ``kAudioHardwarePropertyDevices``
    (what ``SwitchAudioSource`` enumerates) and a higher-level "available routes" list
    that backs the Control Center Sound submenu. A paired BT device whose A2DP profile
    has never engaged on this Mac sits in the second list but not the first — and
    plain ``blueutil --connect`` brings up the BT control link without promoting it.
    The only known way to promote it is to "select" it in the Sound submenu, which
    Apple gates behind a real GUI click.

    Algorithm: open the Sound popover, iterate the AXCheckBox children of its
    scroll area (each is one device row), click each in turn, and after every
    click re-enumerate Core Audio outputs. Stop on first match. On miss,
    restore the prior default (clicking sets the row as default) and return
    False. The caller is then free to ``SwitchAudioSource -s`` the target name.

    Requires Accessibility TCC granted to the daemon that runs dispatched
    commands on ``host_alias``.
    """
    prev_default = await _current_default_output(service, host_alias)
    row_count = await _open_sound_popover_and_count_rows(service, host_alias)
    logger.info('%s: Sound popover open, %d candidate rows — looking for %r', host_alias, row_count, name)
    folded_target = nfkc_casefold(name)
    hit = False
    try:
        for idx in range(1, row_count + 1):
            logger.debug('%s: click checkbox %d/%d', host_alias, idx, row_count)
            await _click_sound_row(service, host_alias, idx)
            await asyncio.sleep(0.4)
            outputs = await _list_core_audio_outputs(service, host_alias)
            if any(nfkc_casefold(o) == folded_target for o in outputs):
                logger.info('%s: rescue hit on row %d → %r in Core Audio', host_alias, idx, name)
                hit = True
                return True
        logger.info('%s: rescue exhausted candidate rows without finding %r', host_alias, name)
        return False
    finally:
        await _close_popover(service, host_alias)
        if not hit and prev_default:
            # Best-effort restore — mirror _close_popover's swallow. Without this
            # guard, a transient dispatch failure during the restore step inside
            # a `finally` triggered by an in-flight exception would mask the
            # original cause (Python finally semantics: new exception replaces
            # the propagating one) and callers branching on BluetoothError.code
            # would see the wrong identifier.
            try:
                await _run(
                    service,
                    host_alias,
                    f'SwitchAudioSource -t output -s {shlex.quote(prev_default)}',
                )
            except BluetoothError:
                logger.exception('%s: failed to restore prior default output %r', host_alias, prev_default)


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
        raise BluetoothError(
            f'{host}: no host results returned from dispatch',
            code='bluetooth-dispatch-no-result',
        )
    hr = result.results[0]
    if hr.error is not None:
        raise BluetoothError(
            f'{host}: dispatch failed: {hr.error}',
            code='bluetooth-dispatch-failed',
        )
    if hr.exit_code != 0:
        stdout = hr.stdout.strip()
        stderr = hr.stderr.strip()
        # The dispatch daemon emits `[TIMEOUT]` to stdout when it kills a child for
        # exceeding the timeout — surface stdout so callers (TCC heuristics, etc.)
        # can distinguish "timed out" from "exited non-zero with empty stderr."
        # 200-char command truncation gives heuristics enough context to detect
        # AppleScript intent (click/keystroke → Accessibility TCC).
        raise BluetoothError(
            f'{host}: command failed (exit {hr.exit_code}): {command[:200]}\n'
            f'  stdout: {stdout or "<empty>"}\n'
            f'  stderr: {stderr or "<empty>"}',
            code='bluetooth-command-failed',
            context={'host': host, 'exit_code': str(hr.exit_code)},
        )
    return hr


def _parse_system_profiler(raw_json: str) -> Sequence[BluetoothDevice]:
    """Parse ``system_profiler SPBluetoothDataType -json`` into a ``BluetoothDevice`` list.

    Returns every paired device the controller reports, including non-audio ones
    (keyboards, trackpads, etc.). Callers narrow by name — and since the names
    that reach this module come from Core Audio output enumeration, the
    narrowing inherently selects audio devices. Filtering on parse would just
    couple us to Apple's evolving minor-type strings for no gain.
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
                out.append(
                    BluetoothDevice(
                        name=name,
                        address=_normalize_address(info.get('device_address', '')),
                        minor_type=info.get('device_minorType', ''),
                        services=info.get('device_services', ''),
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
    folded = nfkc_casefold(name)
    return any(d.connected and nfkc_casefold(d.name) == folded for d in devices)


def _normalize_address(addr: str) -> str:
    """Normalize MAC address to blueutil's canonical form: lowercase, dash-separated."""
    return addr.replace(':', '-').lower()


async def _current_default_output(service: DispatchService, host_alias: str) -> str:
    """Return the current Core Audio default output device name on ``host_alias``."""
    return (await _run(service, host_alias, 'SwitchAudioSource -c -t output')).stdout.strip()


async def _list_core_audio_outputs(service: DispatchService, host_alias: str) -> Sequence[str]:
    """Return non-empty trimmed lines from ``SwitchAudioSource -a -t output`` on ``host_alias``."""
    raw = (await _run(service, host_alias, 'SwitchAudioSource -a -t output')).stdout
    return [line.strip() for line in raw.splitlines() if line.strip()]


_OPEN_AND_COUNT_APPLESCRIPT = """
tell application "System Events"
    tell process "ControlCenter"
        try
            click (first menu bar item of menu bar 1 whose description is "Sound")
        on error errMsg
            return "ERROR_OPEN: " & errMsg
        end try
        delay 0.4
        if (count of windows) = 0 then
            return "ERROR_NOWINDOW"
        end if
        try
            return (count of checkboxes of scroll area 1 of group 1 of window 1) as text
        on error errMsg
            return "ERROR_COUNT: " & errMsg
        end try
    end tell
end tell
"""


async def _open_sound_popover_and_count_rows(service: DispatchService, host_alias: str) -> int:
    """Open the Control Center Sound popover and return the number of device rows.

    Each row is an ``AXCheckBox`` child of the popover's ``AXScrollArea``. Raises
    ``BluetoothError`` with the verbatim AppleScript error when the popover
    can't open (Accessibility TCC denied, ControlCenter not running, Sound icon
    hidden from menu bar, etc.) — the message names what the OS actually said
    rather than guessing at the cause.
    """
    cmd = f'osascript -e {shlex.quote(_OPEN_AND_COUNT_APPLESCRIPT)}'
    result = await _run(service, host_alias, cmd)
    out = result.stdout.strip()
    if out.startswith('ERROR_OPEN'):
        raise BluetoothError(
            f'{host_alias}: Sound popover did not open. AppleScript said: {out}',
            code='bluetooth-sound-menu-popover-not-open',
            context={'host': host_alias, 'applescript_output': out[:200]},
        )
    if out == 'ERROR_NOWINDOW':
        raise BluetoothError(
            f'{host_alias}: Sound popover did not appear after menu bar click',
            code='bluetooth-sound-menu-popover-not-appearing',
            context={'host': host_alias},
        )
    if out.startswith('ERROR_COUNT'):
        raise BluetoothError(
            f'{host_alias}: Sound popover opened but row introspection failed: {out}',
            code='bluetooth-sound-menu-row-introspection-failed',
            context={'host': host_alias, 'applescript_output': out[:200]},
        )
    try:
        return int(out)
    except ValueError as exc:
        # AppleScript exited 0 but stdout isn't one of the three ERROR_* prefixes
        # or a parseable integer. Surface as a structured BluetoothError matching
        # the sibling raise sites above rather than leaking ValueError past the
        # documented BluetoothError contract.
        raise BluetoothError(
            f'{host_alias}: Sound popover row count parse failed: {out!r}',
            code='bluetooth-sound-menu-row-count-parse-failed',
            context={'host': host_alias, 'applescript_output': out[:200]},
        ) from exc


_CLOSE_POPOVER_APPLESCRIPT = 'tell application "System Events" to key code 53'


async def _close_popover(service: DispatchService, host_alias: str) -> None:
    """Dismiss the Sound popover by sending Escape. Best-effort; swallow dispatch errors."""
    try:
        await _run(service, host_alias, f'osascript -e {shlex.quote(_CLOSE_POPOVER_APPLESCRIPT)}')
    except BluetoothError:
        pass


async def _click_sound_row(service: DispatchService, host_alias: str, index: int) -> None:
    """Click the Nth ``AXCheckBox`` row in the open Sound popover.

    Uses AppleScript object reference rather than screen coordinates — robust
    to popover position and size changes across macOS versions and screen
    configurations. The popover must already be open.
    """
    script = (
        'tell application "System Events" to tell process "ControlCenter" '
        f'to click checkbox {index} of scroll area 1 of group 1 of window 1'
    )
    await _run(service, host_alias, f'osascript -e {shlex.quote(script)}')
