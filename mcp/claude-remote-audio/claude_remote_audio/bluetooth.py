from __future__ import annotations

import asyncio
import json
import logging
import shlex
import unicodedata
from collections.abc import Iterator, Sequence

from cc_lib.schemas import ClosedModel
from claude_remote_bash import DispatchService
from claude_remote_bash.dispatch import HostRunResult

__all__ = [
    'BluetoothDevice',
    'BluetoothError',
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

    Algorithm: open the Sound popover, step a click point down each candidate output
    row, and after every click re-enumerate Core Audio outputs. Stop on first match.
    On miss, restore the prior default (the click-walk leaves the last-clicked row
    selected) and return False. The caller is then free to ``SwitchAudioSource -s``
    the target name as the new default.

    Requires Accessibility TCC granted to the daemon that runs dispatched commands
    on ``host_alias``. macOS releases sometimes shift the Sound popover's layout;
    if the rescue starts missing rows it has previously hit, the geometry constants
    in ``_row_click_points`` are the first place to look.
    """
    prev_default = await _current_default_output(service, host_alias)
    bounds = await _open_sound_popover(service, host_alias)
    logger.info('%s: Sound popover open at %s — clicking rows looking for %r', host_alias, bounds, name)
    folded_target = _nfkc_casefold(name)
    hit = False
    try:
        for idx, (click_x, click_y) in enumerate(_row_click_points(bounds)):
            logger.debug('%s: click row %d at (%d, %d)', host_alias, idx, click_x, click_y)
            await _click_at(service, host_alias, click_x, click_y)
            await asyncio.sleep(0.4)
            outputs = await _list_core_audio_outputs(service, host_alias)
            if any(_nfkc_casefold(o) == folded_target for o in outputs):
                logger.info('%s: rescue hit on row %d → %r in Core Audio', host_alias, idx, name)
                hit = True
                return True
        logger.info('%s: rescue exhausted candidate rows without finding %r', host_alias, name)
        return False
    finally:
        await _close_popover(service, host_alias)
        if not hit and prev_default:
            await _run(
                service,
                host_alias,
                f'SwitchAudioSource -t output -s {shlex.quote(prev_default)}',
            )


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


async def _current_default_output(service: DispatchService, host_alias: str) -> str:
    """Return the current Core Audio default output device name on ``host_alias``."""
    return (await _run(service, host_alias, 'SwitchAudioSource -c -t output')).stdout.strip()


async def _list_core_audio_outputs(service: DispatchService, host_alias: str) -> Sequence[str]:
    """Return non-empty trimmed lines from ``SwitchAudioSource -a -t output`` on ``host_alias``."""
    raw = (await _run(service, host_alias, 'SwitchAudioSource -a -t output')).stdout
    return [line.strip() for line in raw.splitlines() if line.strip()]


_OPEN_POPOVER_APPLESCRIPT = """
tell application "System Events"
    tell process "ControlCenter"
        try
            click menu bar item "Sound" of menu bar 1
        on error errMsg
            return "ERROR_OPEN: " & errMsg
        end try
        delay 0.4
        if (count of windows) = 0 then
            return "ERROR_NOWINDOW"
        end if
        set pos to position of window 1
        set sz to size of window 1
        return ((item 1 of pos as integer) as string) & "," & ¬
               ((item 2 of pos as integer) as string) & "," & ¬
               ((item 1 of sz as integer) as string) & "," & ¬
               ((item 2 of sz as integer) as string)
    end tell
end tell
"""


async def _open_sound_popover(service: DispatchService, host_alias: str) -> tuple[int, int, int, int]:
    """Open the Control Center Sound popover and return its (x, y, width, height) on ``host_alias``.

    Raises ``BluetoothError`` if Accessibility TCC isn't granted (caught from
    AppleScript's ``-1719``/``-1743`` error envelope) or the popover doesn't appear.
    """
    cmd = f'osascript -e {shlex.quote(_OPEN_POPOVER_APPLESCRIPT)}'
    result = await _run(service, host_alias, cmd)
    out = result.stdout.strip()
    if out.startswith('ERROR_OPEN'):
        raise BluetoothError(
            f'{host_alias}: cannot open Sound popover — likely Accessibility TCC '
            f'not granted to the claude-remote-bash daemon. Grant via System Settings '
            f'→ Privacy & Security → Accessibility. AppleScript said: {out}'
        )
    if out == 'ERROR_NOWINDOW':
        raise BluetoothError(
            f'{host_alias}: Sound popover did not appear after menu bar click. '
            f'Control Center may be in an unexpected state.'
        )
    parts = out.split(',')
    if len(parts) != 4:
        raise BluetoothError(f'{host_alias}: malformed popover geometry: {out!r}')
    x, y, w, h = (int(p) for p in parts)
    return x, y, w, h


_CLOSE_POPOVER_APPLESCRIPT = 'tell application "System Events" to key code 53'


async def _close_popover(service: DispatchService, host_alias: str) -> None:
    """Dismiss the Sound popover by sending Escape. Best-effort; swallow dispatch errors."""
    try:
        await _run(service, host_alias, f'osascript -e {shlex.quote(_CLOSE_POPOVER_APPLESCRIPT)}')
    except BluetoothError:
        pass


async def _click_at(service: DispatchService, host_alias: str, x: int, y: int) -> None:
    """Synthesize a click at screen coordinates ``(x, y)`` on ``host_alias`` via AppleScript."""
    script = f'tell application "System Events" to click at {{{x}, {y}}}'
    await _run(service, host_alias, f'osascript -e {shlex.quote(script)}')


_POPOVER_TOP_SKIP = 110  # px to skip past Sound title + volume slider + "Output" header
_POPOVER_BOTTOM_SKIP = 50  # px to skip "Sound Settings..." footer
_POPOVER_ROW_HEIGHT = 36  # typical Sound submenu row height on macOS Sequoia


def _row_click_points(bounds: tuple[int, int, int, int]) -> Iterator[tuple[int, int]]:
    """Yield candidate ``(x, y)`` click points for each output row in the Sound popover.

    Geometric heuristic for macOS Sequoia's Control Center Sound submenu — skip the
    header band (Sound title, volume slider, "Output" subheader) and the footer
    button ("Sound Settings..."), step ``_POPOVER_ROW_HEIGHT`` px between candidates,
    and click at horizontal center to avoid the radio-button gutter on the left.
    """
    x, y, width, height = bounds
    click_x = x + width // 2
    cy = y + _POPOVER_TOP_SKIP
    bottom = y + height - _POPOVER_BOTTOM_SKIP
    while cy < bottom:
        yield click_x, cy
        cy += _POPOVER_ROW_HEIGHT
