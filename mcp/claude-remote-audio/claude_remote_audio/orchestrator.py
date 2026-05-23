from __future__ import annotations

import asyncio
import base64
import difflib
import logging
import re
import shlex
import subprocess
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from cc_lib.schemas import ClosedModel
from claude_remote_bash import DispatchService
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.discovery import DiscoveredHost, browse_hosts
from claude_remote_bash.dispatch import HostRunResult
from claude_remote_bash.selector import parse as parse_selector

from claude_remote_audio import bluetooth
from claude_remote_audio.cache import DeviceCache, write_devices
from claude_remote_audio.exceptions import ApplyError, ResolvableApplyError

__all__ = [
    'ApplyResult',
    'HostApplyOutcome',
    'apply',
    'enumerate_devices',
]


_BASE_PORT = 11001
_MIC_RECEIVE_PORT = 10001
_DISPATCH_SESSION = 'claude-remote-audio'
_DISPATCH_TIMEOUT_S = 60.0
_DEVICE_SECTION_SEPARATOR = '---claude-remote-audio-section---'

logger = logging.getLogger(__name__)


# -- Public schemas -----------------------------------------------------------


class HostApplyOutcome(ClosedModel):
    """Result of apply operations against a single host."""

    host: str
    role: Literal['hub', 'peer']
    actions: Sequence[str]
    success: bool
    error: str | None = None


class ApplyResult(ClosedModel):
    """Aggregate apply outcome across every host in ``--target``."""

    hosts: Sequence[HostApplyOutcome]
    overall_success: bool


# -- Public API ---------------------------------------------------------------


async def apply(
    *,
    target: str,
    hub: str | None = None,
    input_device: str | None = None,
    output_device: str | None = None,
    install_prereqs: bool = False,
) -> ApplyResult:
    """Resolve ``target`` into hosts and converge each toward the declared audio topology.

    ``hub=None`` defaults to the local machine — the discovered daemon whose advertised
    IPs overlap with this host's interface IPs (``DiscoveredHost.is_self``). Locality
    model: the command acts on the machine you ran it from. Pass ``hub`` explicitly to
    override.
    """
    logger.info(
        'apply: target=%s hub=%s input=%r output=%r install_prereqs=%s',
        target,
        hub or '<local>',
        input_device,
        output_device,
        install_prereqs,
    )
    service = DispatchService()
    plan = await _build_plan(
        service=service,
        target=target,
        hub=hub,
        input_device=input_device,
        output_device=output_device,
        install_prereqs=install_prereqs,
    )

    outcomes: list[HostApplyOutcome] = []
    if plan.hub_in_target:
        outcomes.append(await _apply_hub(service, plan))
    peer_outcomes = await asyncio.gather(*(_apply_peer(service, plan, peer) for peer in plan.target_peer_aliases))
    outcomes.extend(peer_outcomes)

    return ApplyResult(
        hosts=outcomes,
        overall_success=all(o.success for o in outcomes),
    )


async def enumerate_devices(
    service: DispatchService,
    hub_alias: str,
) -> DeviceCache:
    """Enumerate ``hub_alias``'s Core Audio outputs + inputs in one dispatch; write + return the snapshot.

    The cache write is unconditional on success — every dispatch refreshes the
    on-disk cache backing ``--input`` / ``--output`` tab completion.
    """
    raw = (
        await _run(
            service,
            hub_alias,
            'SwitchAudioSource -a -t output; '
            f"printf '%s\\n' {shlex.quote(_DEVICE_SECTION_SEPARATOR)}; "
            'SwitchAudioSource -a -t input',
        )
    ).stdout
    outputs: list[str] = []
    inputs: list[str] = []
    section = outputs
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if line == _DEVICE_SECTION_SEPARATOR:
            section = inputs
            continue
        if line:
            section.append(line)
    return write_devices(hub_alias, outputs=outputs, inputs=inputs)


# -- Plan internals -----------------------------------------------------------


class _Plan(ClosedModel):
    """Resolved topology — what every phase reads from.

    Two peer sets:

    - **Topology**: every discovered Mac except hub. Drives Phase B's roc-send
      destinations and roc-recv port bindings — the hub broadcasts to and listens
      from every peer in the topology, regardless of ``--target``.
    - **Target**: peers in ``--target``. Drives Phase C device mutations. Always a
      subset of topology.
    """

    hub_alias: str
    hub_in_target: bool
    hub_ip: str
    hub_loopback_port: int | None
    hub_outputs: Sequence[str]
    hub_inputs: Sequence[str]
    topology_peer_aliases: Sequence[str]
    topology_peer_to_ip: Mapping[str, str]
    topology_peer_to_port: Mapping[str, int]
    target_peer_aliases: Sequence[str]
    input_device: str | None
    output_device: str | None


async def _build_plan(
    *,
    service: DispatchService,
    target: str,
    hub: str | None,
    input_device: str | None,
    output_device: str | None,
    install_prereqs: bool,
) -> _Plan:
    """Parse ``--target`` against mDNS discovery + client groups, resolve hub default if absent."""
    browse = await browse_hosts(timeout=3.0)
    all_hosts = [*browse.remote_daemons, *([browse.local_daemon] if browse.local_daemon else [])]
    aliases: dict[str, DiscoveredHost] = {h.alias.lower(): h for h in all_hosts}
    if not aliases:
        raise ApplyError('no daemons discovered on the LAN — ensure claude-remote-bash-daemon is running on each Mac')

    if hub is None:
        if browse.local_daemon is None:
            raise ApplyError(
                'no --hub provided and the local machine is not a discovered daemon. '
                f'Discovered: {sorted(aliases.keys())}. '
                'Start claude-remote-bash-daemon locally or pass --hub explicitly.'
            )
        hub = browse.local_daemon.alias

    config = ClientConfig.load()
    atoms = parse_selector(
        target,
        groups=config.groups,
        discovered_aliases=set(aliases.keys()),
    )
    atom_lower = [a.lower() for a in atoms]
    hub_lower = hub.lower()

    for atom in atom_lower:
        if ':' in atom:
            raise ApplyError(
                f'--target atom {atom!r} is an ip:port literal — not supported. '
                'Use the host alias from `claude-remote-bash discover` instead.'
            )

    if hub_lower not in aliases:
        raise ApplyError(f'--hub {hub!r} is not discoverable on the LAN')

    hub_in_target = hub_lower in atom_lower
    hub_host = aliases[hub_lower]
    hub_ip = _best_ip(hub_host)

    target_peer_aliases = sorted(a for a in atom_lower if a != hub_lower)
    for p in target_peer_aliases:
        if p not in aliases:
            raise ApplyError(f'peer {p!r} resolved from target but not discoverable')

    await _ensure_prereqs(
        service=service,
        hub_alias=hub,
        target_peer_aliases=target_peer_aliases,
        has_output=output_device is not None,
        has_input=input_device is not None,
        install_prereqs=install_prereqs,
    )

    if output_device is not None:
        mesh_aliases = sorted(v.alias for v in aliases.values())
        await _ensure_bluetooth_output(service, aliases[hub_lower].alias, mesh_aliases, output_device)

    hub_devices = await enumerate_devices(service, hub)
    hub_loopback_port = await _read_hub_loopback_port(service, hub)
    topology_peer_aliases = sorted(a for a in aliases if a != hub_lower)
    topology_peer_to_ip = {p: _best_ip(aliases[p]) for p in topology_peer_aliases}
    topology_peer_to_port = await _assign_peer_ports(service, topology_peer_aliases, hub_ip)

    return _Plan(
        hub_alias=hub,
        hub_in_target=hub_in_target,
        hub_ip=hub_ip,
        hub_loopback_port=hub_loopback_port,
        hub_outputs=hub_devices.outputs,
        hub_inputs=hub_devices.inputs,
        topology_peer_aliases=topology_peer_aliases,
        topology_peer_to_ip=topology_peer_to_ip,
        topology_peer_to_port=topology_peer_to_port,
        target_peer_aliases=target_peer_aliases,
        input_device=input_device,
        output_device=output_device,
    )


def _best_ip(host: DiscoveredHost) -> str:
    """Pick a single IPv4 address from a host's mDNS-advertised set.

    Addresses arrive pre-sorted by ``_address_rank`` (Ethernet first, then
    Wi-Fi, etc.), so the first element is the preferred path.
    """
    if not host.addresses:
        raise ApplyError(f'host {host.alias!r} has no advertised addresses')
    return host.addresses[0].ip


async def _assign_peer_ports(
    service: DispatchService,
    peer_aliases: Sequence[str],
    hub_ip: str,
) -> Mapping[str, int]:
    """Assign per-peer Flow-2 ports — preserve existing peer-Speaker ports, then alpha-fill gaps."""
    used: set[int] = set()
    port_map: dict[str, int] = {}

    for peer in peer_aliases:
        existing = await _read_existing_peer_port(service, peer, hub_ip)
        if existing is not None and existing not in used:
            port_map[peer] = existing
            used.add(existing)

    next_port = _BASE_PORT
    for peer in peer_aliases:
        if peer in port_map:
            continue
        while next_port in used:
            next_port += 1
        port_map[peer] = next_port
        used.add(next_port)
        next_port += 1

    return port_map


async def _ensure_bluetooth_output(
    service: DispatchService,
    hub_alias: str,
    mesh_aliases: Sequence[str],
    output_device: str,
) -> None:
    """If ``output_device`` is a paired Bluetooth audio device, claim it for the hub.

    Called from ``_build_plan`` *before* ``enumerate_devices`` so the freshly-claimed
    device appears in Core Audio's output list when we enumerate. No-op when the
    device is wired / AirPlay / virtual / not paired on the hub — the subsequent
    ``_resolve_output_device`` step will surface a clear "not found" error if the
    device doesn't show up after this step.

    Auto-steal: when the device is paired on the hub but not currently connected,
    we use ``bluetooth.steal`` which also disconnects on any *other* mesh host
    that has it. This is necessary because macOS Continuity allows multi-link
    sharing (paired BT audio appears "connected" on multiple iCloud-linked Macs)
    — without the disconnect-on-source step, audio mixes from every Mac that has
    it instead of routing exclusively to the hub.

    Cold-state rescue: a BT device whose A2DP profile has never engaged on this
    hub stays in the Sound submenu's "available routes" list without being
    promoted into Core Audio's device enumeration; plain ``blueutil --connect``
    doesn't promote it either. When we detect that state (BT link up, but
    ``SwitchAudioSource`` doesn't list the device), we call
    ``bluetooth.engage_via_sound_menu`` to click the device's row in the
    Control Center Sound popover. After one successful rescue, Core Audio
    remembers the device and subsequent applies hit the warm path without
    re-rescuing.
    """
    logger.info('%s: ensuring Bluetooth output %r', hub_alias, output_device)
    try:
        bt_device = await bluetooth.find_device(service, hub_alias, output_device)
    except bluetooth.BluetoothError as exc:
        raise ApplyError(f'{hub_alias}: bluetooth probe failed: {exc}') from exc

    if bt_device is None:
        logger.info('%s: %r is not a paired BT device — deferring to Core Audio resolution', hub_alias, output_device)
        return

    if not bt_device.connected:
        try:
            await bluetooth.steal(service, hub_alias, bt_device.address, mesh_aliases)
        except bluetooth.BluetoothError as exc:
            if _looks_like_bluetooth_tcc_denial(str(exc)):
                raise ResolvableApplyError(
                    f'{hub_alias}: Bluetooth control blocked — the dispatching app on '
                    f'{hub_alias} has not been granted Bluetooth permission.',
                    code='bluetooth-tcc-denied',
                    title='Bluetooth permission required',
                    suggestions=(
                        f'On {hub_alias}: click Allow on the macOS dialog if visible '
                        f'(the dispatching app — iTerm, Terminal, or your IDE — needs '
                        f'Bluetooth access).',
                        'If no dialog or already dismissed: System Settings → Privacy '
                        "& Security → Bluetooth → enable the dispatching app's entry.",
                        'Re-run apply after granting permission. The grant persists.',
                    ),
                    context={'host': hub_alias, 'device': bt_device.name},
                ) from exc
            raise ApplyError(f'{hub_alias}: failed to claim Bluetooth device {bt_device.name!r}: {exc}') from exc
        # A2DP engagement / Core Audio registration lags the BT control link by
        # ~1-3s. Wait so the verification below sees the warm path before we
        # decide to invoke the GUI rescue.
        await asyncio.sleep(2)

    if await _device_in_hub_outputs(service, hub_alias, bt_device.name):
        logger.info('%s: %r is engaged in Core Audio (warm path)', hub_alias, bt_device.name)
        return

    logger.info('%s: %r BT link-up but absent from Core Audio — Sound-menu rescue', hub_alias, bt_device.name)
    try:
        engaged = await bluetooth.engage_via_sound_menu(service, hub_alias, bt_device.name)
    except bluetooth.BluetoothError as exc:
        if _looks_like_applescript_tcc_denial(str(exc)):
            raise ResolvableApplyError(
                f'{hub_alias}: Sound-menu rescue blocked by AppleScript automation permission.',
                code='applescript-system-events-denied',
                title='AppleScript automation permission required',
                suggestions=(
                    f'On {hub_alias}: click Allow on the macOS dialog that appeared '
                    f'(the dispatching app — typically iTerm, Terminal, or your IDE — '
                    f'needs to control "System Events").',
                    f'If no dialog is visible on {hub_alias}: System Settings → Privacy '
                    "& Security → Automation → enable the dispatching app's entry for "
                    '"System Events".',
                    'Re-run apply after granting permission. The grant persists.',
                ),
                context={'host': hub_alias, 'device': bt_device.name},
            ) from exc
        raise ApplyError(f'{hub_alias}: Sound-menu rescue failed for {bt_device.name!r}: {exc}') from exc

    if not engaged:
        raise ApplyError(
            f'{hub_alias}: Bluetooth link is up to {bt_device.name!r} but Core Audio '
            f'will not enumerate it, and the Sound-menu rescue did not find it among '
            f'the popover rows. Check that the device is powered on and visible in '
            f'System Settings → Sound on {hub_alias!r}.'
        )
    logger.info('%s: rescue engaged %r in Core Audio', hub_alias, bt_device.name)


async def _device_in_hub_outputs(service: DispatchService, hub_alias: str, name: str) -> bool:
    """True if ``name`` (smart-quote-folded) appears in ``SwitchAudioSource -a -t output``."""
    raw = (await _run(service, hub_alias, 'SwitchAudioSource -a -t output')).stdout
    folded = _nfkc_casefold(name)
    return any(_nfkc_casefold(line.strip()) == folded for line in raw.splitlines() if line.strip())


def _looks_like_applescript_tcc_denial(error_msg: str) -> bool:
    """True when a BluetoothError looks like an AppleScript Automation TCC denial.

    Signal: a non-zero osascript exit on a command that tells ``"System Events"``
    while the user hasn't (yet) granted Automation permission. macOS often
    suppresses osascript's stderr in this state (the process is held while the
    permission dialog awaits user action; the daemon-side dispatch timeout fires
    and reports exit -1 with empty stderr). We match on the osascript + System
    Events pair regardless of exit code or stderr content — far more often
    that combination is TCC denial than something else.
    """
    if 'osascript' not in error_msg:
        return False
    if 'System Events' not in error_msg:
        return False
    return 'command failed (exit' in error_msg


def _looks_like_bluetooth_tcc_denial(error_msg: str) -> bool:
    """True when a BluetoothError matches blueutil's documented TCC denial signal.

    Only deterministic markers — no inference from timeout signatures, which
    are indistinguishable from device-unreachable failures (Apple's BT layer
    waits ~30s on unreachable connects; the daemon-side dispatch timeout
    fires first with the same ``exit -1 + empty stderr`` signature as a TCC
    pending-dialog hold). A previous version's third clause matching that
    pair caused false positives for "device not reachable" errors, sending
    users to grant permissions they already had.

    Two clauses, both load-bearing on blueutil v2.13.0's source behavior:

    - **Already denied**: blueutil installs a ``SIGABRT`` handler
      (``handle_abort`` in ``blueutil.m``) that runs when CoreBluetooth aborts
      on a permission failure. It writes ``"Error: Received abort signal, it
      may be due to absence of access to Bluetooth API..."`` to stderr and
      exits with ``EX_SIGABRT`` (134). No other failure mode in blueutil
      produces exit 134.

    For pending-dialog state (rarer — the user is mid-dialog), the failure
    surfaces as a generic ApplyError with the underlying timeout text. Less
    actionable than a structured error, but not WRONG.
    """
    if 'blueutil' not in error_msg:
        return False
    if 'absence of access to Bluetooth API' in error_msg:
        return True
    return 'command failed (exit 134)' in error_msg


# -- Hub phase ----------------------------------------------------------------


async def _apply_hub(service: DispatchService, plan: _Plan) -> HostApplyOutcome:
    """Run hub-side mutations: feedback-loop guard, then opt-in output and input management."""
    await _guard_feedback_loop(service, plan)
    actions: list[str] = []
    if plan.output_device is not None:
        actions.extend(await _set_hub_output(service, plan))
        actions.extend(await _restart_roc_recv(service, plan))
    if plan.input_device is not None:
        hub_devices = (await _run(service, plan.hub_alias, 'roc-vad device list')).stdout
        actions.extend(await _ensure_claude_remote_mic(service, plan.hub_alias, hub_devices))
        actions.extend(await _restart_roc_send(service, plan))
        actions.extend(await _set_hub_input_to_remote_mic(service, plan))
    return HostApplyOutcome(host=plan.hub_alias, role='hub', actions=actions, success=True)


async def _set_hub_input_to_remote_mic(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Set the hub's Core Audio default input to ``Claude Remote Mic`` at full volume.

    Apps on the hub then consume the mesh-distributed mic via the self-loopback
    leg of roc-send (127.0.0.1:10001 → Claude Remote Mic), giving consistent
    behavior across the entire mesh: every Mac's "default mic" is the hub's
    physical input. Volume pinned to max — gain belongs at the physical mic,
    not the digital intermediate.
    """
    await _run(service, plan.hub_alias, 'SwitchAudioSource -t input -s "Claude Remote Mic"')
    await _run(service, plan.hub_alias, 'claude-coreaudio-volume set "Claude Remote Mic" input 1.0')
    return ['set default input → Claude Remote Mic (volume → max)']


async def _guard_feedback_loop(service: DispatchService, plan: _Plan) -> None:
    """Refuse to mutate if the hub default output would create a feedback loop via Claude Remote Speaker."""
    if plan.output_device == 'Claude Remote Speaker':
        raise ApplyError(
            '--output cannot be "Claude Remote Speaker" — that creates a feedback loop '
            '(hub default output → Claude Remote Speaker sender → ships back to hub roc-recv → '
            'plays into default output → infinite amplification).'
        )

    current_output = (await _run(service, plan.hub_alias, 'SwitchAudioSource -c -t output')).stdout.strip()
    if current_output == 'Claude Remote Speaker' and plan.output_device is None:
        raise ApplyError(
            'hub default output is currently "Claude Remote Speaker" — feedback risk. '
            'Pass --output <safe-device> to switch first.'
        )


async def _set_hub_output(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Resolve ``--output`` against the canonical list; set hub default + system-effects output."""
    assert plan.output_device is not None  # guaranteed by caller
    canonical = _resolve_output_device(plan.hub_outputs, plan.hub_alias, plan.output_device)
    quoted = shlex.quote(canonical)
    await _run(service, plan.hub_alias, f'SwitchAudioSource -t output -s {quoted}')
    await _run(service, plan.hub_alias, f'SwitchAudioSource -t system -s {quoted}')
    return [f'set default output + system output → {canonical}']


def _resolve_output_device(canonical: Sequence[str], hub_alias: str, requested: str) -> str:
    """Match ``requested`` against the pre-fetched canonical output list.

    Strategy:

    1. Exact match.
    2. NFKC case-folded match with smart-quote / NBSP folding (U+2019 → U+0027,
       U+201C/D → U+0022, U+00A0 → ASCII space). Safe because the matched
       canonical name is what gets passed to ``SwitchAudioSource -s``; lenient
       matching only widens resolution, never execution.
    3. Miss → raise ``ApplyError`` with a ``difflib.get_close_matches`` "did you mean" hint.
    """
    if requested in canonical:
        return requested

    folded_req = _nfkc_casefold(requested)
    matches = [c for c in canonical if _nfkc_casefold(c) == folded_req]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ApplyError(f'{hub_alias}: --output {requested!r} ambiguous: {matches}')

    suggestion = difflib.get_close_matches(requested, list(canonical), n=1, cutoff=0.6)
    hint = f' Did you mean {suggestion[0]!r}?' if suggestion else ''
    raise ApplyError(f'{hub_alias}: --output {requested!r} not found. Available: {list(canonical)}.{hint}')


_QUOTE_FOLD_TABLE = str.maketrans(
    {
        '\u2018': "'",  # left single quote → ASCII apostrophe
        '\u2019': "'",  # right single quote / apostrophe — what Core Audio uses for Chris's AirPods Max
        '\u201c': '"',  # left double quote → ASCII double quote
        '\u201d': '"',  # right double quote → ASCII double quote
        '\u00a0': ' ',  # NBSP → ASCII space
    }
)


def _nfkc_casefold(s: str) -> str:
    """NFKC-normalize, fold smart quotes and NBSP to ASCII, then case-fold.

    The smart-quote fold is the load-bearing piece: Core Audio uses U+2019
    (curly apostrophe) in user-renamed devices like ``Chris's AirPods Max``,
    but keyboards type U+0027 (straight). NFKC alone does not bridge those —
    they are different semantic characters, not Unicode-equivalence variants.
    Folding is safe because callers always pass the *canonical* matched name
    to destructive commands; the fold only widens matching, not side effects.
    """
    return unicodedata.normalize('NFKC', s).translate(_QUOTE_FOLD_TABLE).casefold()


async def _restart_roc_recv(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Kill any running roc-recv on the hub; relaunch bound to all topology peer ports + self-loopback.

    Preflights the Application Firewall for roc-recv — when it's blocked or
    not-listed, the firewall pops a GUI prompt the daemon chain can't
    reliably get clicked, and incoming UDP packets get silently dropped.
    Same shape as the TCC preflight in `_restart_roc_send`.
    """
    recv_fw = await _check_application_firewall(service, plan.hub_alias, '/usr/local/bin/roc-recv')
    if recv_fw in ('blocked', 'not-listed'):
        raise ResolvableApplyError(
            f'{plan.hub_alias}: macOS Application Firewall blocks /usr/local/bin/roc-recv '
            f'(status={recv_fw}). Incoming UDP packets from mesh peers would be silently dropped.',
            code='roc-recv-firewall-blocked',
            title=f'roc-recv blocked by Application Firewall on {plan.hub_alias}',
            suggestions=(
                f'On {plan.hub_alias}, pre-authorize via:',
                '  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/roc-recv',
                '  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/roc-recv',
                'Then re-run apply. (Alternative: open System Settings → Network → Firewall '
                '→ Options, find roc-recv, set to Allow.)',
            ),
            context={'host': plan.hub_alias, 'binary': '/usr/local/bin/roc-recv', 'status': recv_fw},
        )
    bind_ports = [plan.topology_peer_to_port[p] for p in plan.topology_peer_aliases]
    if plan.hub_loopback_port is not None:
        bind_ports.append(plan.hub_loopback_port)
    await _run(service, plan.hub_alias, _roc_recv_command(bind_ports))
    return ['restarted roc-recv']


# SoX's deprecated ``coreaudio.c`` reads device names via the C-string variant
# of ``kAudioDevicePropertyDeviceName`` into a fixed-size buffer. Names ≤ this
# many UTF-8 bytes survive enumeration intact; longer names get truncated
# mid-string and either fail lookup or collide with other devices sharing the
# truncated prefix. Empirical data points (verbose ``roc-send`` enums on M4/M2):
#
#   - ``BenQ PD3200U`` (12 bytes)          → shown in full   ✓
#   - ``Kuycon G32P`` (11 bytes)           → shown in full   ✓
#   - ``DJI MIC MINI`` (12 bytes)          → known-working in M5 reference apply ✓
#   - ``Claude Remote Mic`` (17 bytes)     → truncated to ``"Claude Remo"``
#   - ``Samson Q2U Microphone`` (21 bytes) → truncated to ``"Samson Q2U "``
#
# So the boundary is 12 bytes (with NUL terminator that's a 13-byte buffer).
# Used only by the truncation preflight below; kept near its consumer rather
# than with the network/dispatch knobs at module top. sox_ng fixes the
# underlying API choice — see task #52.
_SOX_DEVICE_NAME_MAX_BYTES = 12


async def _restart_roc_send(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Kill any running roc-send on the hub; relaunch with direct ``core://`` capture.

    Preflights Microphone TCC before spawning roc-send. macOS blocks daemon-
    spawned mic-touching processes on the TCC prompt — they stay *alive* but
    do nothing useful, fooling the post-launch ``pgrep`` survival check into
    reporting success against a stuck process (observed empirically on
    2026-05-21). Catching the ``notDetermined`` / ``denied`` state at preflight
    avoids creating the zombie entirely.

    Spawns roc-send detached via ``nohup`` so the dispatch returns immediately
    even when the child crashes on startup (bad device name, channel-count
    mismatch, etc.). Verifies survival via post-launch pgrep. On failure, runs
    the multi-tier diagnostic so the resulting ``ResolvableApplyError`` carries
    a stable ``code`` + actionable ``suggestions`` — consumers (humans, agents,
    log pipelines) dispatch on the structure rather than parsing prose.
    """
    assert plan.input_device is not None  # guaranteed by caller

    # Preflight: macOS Application Firewall. Same shape as TCC — a permission
    # gate that surfaces as a GUI prompt and silently blocks until granted. If
    # the binary isn't already permitted, daemon-spawned children can't get
    # the prompt clicked in time. Refuse here rather than letting roc-send
    # appear alive while its outbound UDP packets get dropped.
    send_fw = await _check_application_firewall(service, plan.hub_alias, '/usr/local/bin/roc-send')
    if send_fw in ('blocked', 'not-listed'):
        raise ResolvableApplyError(
            f'{plan.hub_alias}: macOS Application Firewall blocks /usr/local/bin/roc-send '
            f'(status={send_fw}). Outbound UDP packets to mesh peers would be silently dropped.',
            code='roc-send-firewall-blocked',
            title=f'roc-send blocked by Application Firewall on {plan.hub_alias}',
            suggestions=(
                f'On {plan.hub_alias}, pre-authorize via:',
                '  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/roc-send',
                '  sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/roc-send',
                'Then re-run apply. (Alternative: open System Settings → Network → Firewall '
                '→ Options, find roc-send, set to Allow.)',
            ),
            context={'host': plan.hub_alias, 'binary': '/usr/local/bin/roc-send', 'status': send_fw},
        )

    # Preflight: Microphone TCC. notDetermined means "never asked" — instead of
    # refusing and making the user manually run sox-d-n to trigger the prompt,
    # we ask the hub to trigger it itself via `claude-tcc-probe --request`
    # (wraps `AVCaptureDevice.requestAccess(.audio)`). The macOS prompt pops on
    # the hub's screen; user clicks; the dispatch returns the resolved status.
    # Denied / restricted are NOT auto-recoverable from CLI (require Settings
    # toggle on / MDM policy change) — refuse in both those cases.
    mic_status = await _check_microphone_tcc(service, plan.hub_alias)
    if mic_status == 'notDetermined':
        logger.info(
            '%s: Microphone TCC notDetermined — triggering prompt on hub (click Allow when it appears)',
            plan.hub_alias,
        )
        mic_status = await _request_microphone_tcc(service, plan.hub_alias)
        logger.info('%s: Microphone TCC request resolved: %s', plan.hub_alias, mic_status)
    if mic_status in ('denied', 'restricted'):
        code, title, diagnosis_msg, suggestions = await _diagnose_roc_send_start_failure(
            service, plan.hub_alias, roc_send_log=''
        )
        raise ResolvableApplyError(
            f'{plan.hub_alias}: Microphone TCC preflight failed (status={mic_status}) — '
            f'aborting before roc-send launch to avoid a TCC-blocked zombie process.\n\n'
            f'{diagnosis_msg}',
            code=code,
            title=title,
            suggestions=suggestions,
            context={'host': plan.hub_alias, 'input_device': plan.input_device, 'tcc_status': mic_status},
        )

    # Preflight: input device sanity checks (SoX-CoreAudio shortcomings that
    # the post-mortem diagnostic would otherwise catch as generic open failures).
    # Failing fast here gives more direct error messages, avoids wasted roc-send
    # launches, and handles cases before /tmp/roc-send.log is even written.

    # (z) SoX truncates device names at _SOX_DEVICE_NAME_MAX_BYTES during its
    # CoreAudio enumeration (SoX bug #354 — see constant docstring above for
    # the empirical data points). Longer names lose their tail mid-string;
    # the resulting prefix may not match any device or may accidentally match
    # a different device sharing the truncated prefix. Refuse longer names
    # here rather than letting roc-send fail generically.
    input_bytes = len(plan.input_device.encode('utf-8'))
    if input_bytes > _SOX_DEVICE_NAME_MAX_BYTES:
        raise ResolvableApplyError(
            f'{plan.hub_alias}: input device {plan.input_device!r} is {input_bytes} UTF-8 bytes; '
            f"SoX's deprecated CoreAudio backend truncates device names at "
            f'{_SOX_DEVICE_NAME_MAX_BYTES} bytes during enumeration (SoX bug #354). '
            'Longer names fail to match any device — or worse, accidentally match a '
            'different device sharing the truncated prefix.',
            code='input-device-name-too-long-for-sox',
            title=f'Input device name too long for SoX on {plan.hub_alias}',
            suggestions=(
                f'On {plan.hub_alias}: rename the device (via Audio MIDI Setup → Aggregate '
                f'Device) to ≤{_SOX_DEVICE_NAME_MAX_BYTES} UTF-8 bytes with a unique prefix '
                'no other device shares.',
                'Safe examples: "DJI MIC MINI" (12), "Q2U Mic" (7), "Hub Mic" (7), "CRA Mic" (7), "Mic" (3).',
                f"Avoid names starting with any existing device's first {_SOX_DEVICE_NAME_MAX_BYTES} "
                'bytes (run `claude-coreaudio-volume list` to inventory).',
                'This bug is fixed in sox_ng (the modernized SoX fork) — see task #52 for adoption work.',
            ),
            context={
                'host': plan.hub_alias,
                'input_device': plan.input_device,
                'input_bytes': str(input_bytes),
            },
        )

    match_count, in_channels = await _probe_input_device(service, plan.hub_alias, plan.input_device)

    # (a) Name collision — same name applied to multiple Core Audio devices
    # (USB mics with built-in monitoring expose input + output as separate
    # devices sharing the display name). SoX's deprecated CoreAudio backend
    # has no scope filter; it picks the first match and fails. The clean
    # user-facing workaround is to create an AMS Aggregate Device wrapping
    # only the input substream and give it a unique name. (Physical-device
    # names are immutable in AMS — only aggregate-device names are editable.)
    if match_count >= 2:
        raise ResolvableApplyError(
            f'{plan.hub_alias}: input device {plan.input_device!r} matches {match_count} distinct '
            "Core Audio devices with the same name. SoX's deprecated CoreAudio backend cannot "
            'disambiguate by scope — it picks the first match (often the output side) and fails.',
            code='input-device-name-collision',
            title=f'Same-named input devices on {plan.hub_alias}',
            suggestions=(
                f'On {plan.hub_alias}: run `claude-coreaudio-volume list` to see all matching entries',
                'Open Audio MIDI Setup.app (in /Applications/Utilities/) on that host',
                f'Create a new Aggregate Device (+ button) containing ONLY the input substream of {plan.input_device!r}',
                'Give the aggregate a unique name (e.g. "Samson Q2U Mic"); apply will then open that '
                'unique name without the SoX collision (physical-device names are immutable in macOS; '
                'aggregates are the official disambiguation mechanism)',
                'Re-run apply with `--input "<your unique aggregate name>"`',
            ),
            context={
                'host': plan.hub_alias,
                'input_device': plan.input_device,
                'match_count': str(match_count),
            },
        )

    # (b) Mono input — roc-send always requests 2 channels and has no
    # negotiation flag, so we wrap mono devices in a sox stdin pipe that
    # upmixes mono → stereo before roc-send sees the stream. See
    # `_roc_send_command` for the pipeline details + buffer sizing.
    # `mono = True` when the probe found 0 or 1 input channels (or didn't
    # find the device at all — fall through to letting roc-send fail loudly
    # with a real error rather than guessing).
    mono = in_channels is not None and in_channels < 2

    peer_ips = [plan.topology_peer_to_ip[p] for p in plan.topology_peer_aliases]
    await _run(service, plan.hub_alias, _roc_send_command(plan.input_device, peer_ips, mono=mono))
    await asyncio.sleep(3)
    alive = await _run(service, plan.hub_alias, 'pgrep -f roc-send >/dev/null && echo alive || echo dead')
    if alive.stdout.strip() != 'alive':
        logs = await _run(service, plan.hub_alias, 'tail -20 /tmp/roc-send.log 2>/dev/null')
        code, title, diagnosis_msg, suggestions = await _diagnose_roc_send_start_failure(
            service, plan.hub_alias, roc_send_log=logs.stdout
        )
        raise ResolvableApplyError(
            f'{plan.hub_alias}: roc-send died within 3s of launch — input={plan.input_device!r}.\n\n'
            f'{diagnosis_msg}\n\n'
            f'--- /tmp/roc-send.log ---\n{logs.stdout}',
            code=code,
            title=title,
            suggestions=suggestions,
            context={'host': plan.hub_alias, 'input_device': plan.input_device},
        )
    return [f'restarted roc-send (input={plan.input_device})']


async def _check_application_firewall(service: DispatchService, host: str, binary_path: str) -> str:
    """Return macOS Application Firewall status for ``binary_path`` on ``host``.

    macOS' Application Firewall is a SEPARATE permission gate from TCC. When
    a binary not previously classified by the firewall tries to accept an
    incoming connection, macOS pops a GUI prompt — "Do you want X to accept
    incoming network connections?". For daemon-spawned processes (our chain),
    the prompt may surface to the user-session app (iTerm); if the user
    doesn't see/click it, the connection is silently blocked. Same shape as
    the TCC failure modes we already detect — different mechanism, identical
    consequence: ``pgrep`` reports alive while packets are dropped.

    Uses ``/usr/libexec/ApplicationFirewall/socketfilterfw`` which (verified
    empirically 2026-05-21 on M4) accepts query operations at non-root uid.

    Returns one of:

    - ``firewall-disabled`` — global firewall off; no prompts/blocks possible
    - ``permitted`` — binary explicitly allowed; safe to proceed
    - ``blocked`` — binary explicitly denied; refuse with helpful error
    - ``not-listed`` — binary not yet classified; FIRST incoming connection
      triggers a GUI prompt. Refuse — daemon-spawned processes can't reliably
      get the prompt clicked in time.
    - ``unknown`` — socketfilterfw query failed or returned unexpected text;
      caller treats as "couldn't determine, proceed cautiously."
    """
    state = await _run(service, host, '/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null')
    if 'state = 0' in state.stdout.lower() or 'disabled' in state.stdout.lower():
        return 'firewall-disabled'
    result = await _run(
        service,
        host,
        f'/usr/libexec/ApplicationFirewall/socketfilterfw --getappblocked {shlex.quote(binary_path)} 2>/dev/null',
    )
    output = result.stdout.lower()
    if 'permitted' in output:
        return 'permitted'
    if 'blocked' in output:
        return 'blocked'
    if 'not listed' in output or 'not in the firewall' in output:
        return 'not-listed'
    return 'unknown'


async def _check_microphone_tcc(service: DispatchService, host: str) -> str:
    """Return the responsible-app's Microphone TCC status on ``host``.

    Runs ``claude-tcc-probe microphone`` — a tiny stdlib-only Swift CLI that
    wraps ``AVCaptureDevice.authorizationStatus(for: .audio)``. The probe runs
    in the same dispatch chain as ``roc-send``, so the authorization status
    reflects whatever TCC subject our daemon-spawned grandchildren inherit
    (typically the parent terminal app when the daemon is ``nohup``-launched
    from a shell).

    Returns one of ``authorized`` / ``denied`` / ``notDetermined`` /
    ``restricted`` / ``unknown``. ``unknown`` if the probe binary is missing
    (not yet bootstrapped) or the probe fails for any other reason — the
    caller treats that as "couldn't determine, fall through to HAL probe."
    """
    result = await _run(service, host, '/usr/local/bin/claude-tcc-probe microphone 2>/dev/null || echo unknown')
    return result.stdout.strip().split('\n')[-1] or 'unknown'


async def _request_microphone_tcc(service: DispatchService, host: str) -> str:
    """Trigger Microphone TCC prompt on ``host``, block until user resolves, return final status.

    Runs ``claude-tcc-probe microphone --request`` which wraps
    ``AVCaptureDevice.requestAccess(.audio)``. When the current state is
    ``notDetermined`` macOS shows the system Microphone prompt on the hub's
    screen and the binary blocks on a ``DispatchSemaphore`` until the user
    clicks Allow / Don't Allow; for any other state the request callback
    fires immediately with the cached result.

    Dispatch timeout is raised to 5 minutes — long enough for a user to walk
    over and click without the daemon-side timeout firing while they're
    deciding. If they don't respond within that window the dispatch fails
    upstream and the caller can re-run apply to re-trigger the prompt.
    """
    result = await service.run_target(
        host,
        '/usr/local/bin/claude-tcc-probe microphone --request 2>/dev/null || echo unknown',
        session_id=_DISPATCH_SESSION,
        agent_id=None,
        timeout=300.0,
    )
    if not result.results:
        return 'unknown'
    return result.results[0].stdout.strip().split('\n')[-1] or 'unknown'


async def _probe_input_device(service: DispatchService, host: str, device_name: str) -> tuple[int, int | None]:
    """Inventory same-named CoreAudio devices for ``device_name`` on ``host``.

    Returns ``(match_count, input_channels)``:

    - ``match_count`` is the total number of Core Audio devices on the hub
      whose ``kAudioDevicePropertyDeviceNameCFString`` matches the requested
      name. USB audio interfaces with built-in headphone monitoring commonly
      expose two devices with the same display name — one for the mic input
      scope, one for the headphone output scope — so ``match_count >= 2`` is
      a name-collision signal that SoX's deprecated CoreAudio backend cannot
      disambiguate (it picks the first match without scope-filtering).
    - ``input_channels`` is the native channel count of the first matching
      device with a non-zero input scope, or ``None`` when no matching device
      has an input scope at all (e.g., the user passed an output-only device).

    Both fields independently let the caller refuse with different
    ResolvableApplyError codes (collision vs mono).

    Parses ``claude-coreaudio-volume list`` output. Each line:
    ``id=N  in=Xch vol=Y  out=Zch vol=W  name=DEVICE NAME``. Counts every
    matching name (collision signal) AND finds the input channel count of
    the first input-scoped match. Both fields independently let the caller
    refuse with different ResolvableApplyError codes (collision vs mono).
    """
    result = await _run(service, host, 'claude-coreaudio-volume list 2>/dev/null')
    pattern = re.compile(r'id=\d+\s+in=(\d+)ch\s+vol=\S+\s+out=\d+ch\s+vol=\S+\s+name=(.+?)\s*$')
    match_count = 0
    input_channels: int | None = None
    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        in_channels, name = int(match.group(1)), match.group(2)
        if name == device_name:
            match_count += 1
            if input_channels is None and in_channels > 0:
                input_channels = in_channels
    return match_count, input_channels


async def _diagnose_roc_send_start_failure(
    service: DispatchService,
    host: str,
    *,
    roc_send_log: str = '',
) -> tuple[str, str, str, Sequence[str]]:
    """Diagnose why ``roc-send`` couldn't start on ``host`` in failure-likelihood order.

    SoX wraps every input-open error as a generic ``roc_sndio: backend
    dispatcher: failed to open source`` — TCC denial, HAL wedge, channel-count
    mismatch, and device-specific issues all produce that wrapper line. The
    libsox layer below sometimes leaves a more specific message (``sox source:
    can't open input file or device with the requested channel count`` etc.)
    just above it. This function walks four tiers, each cheap, each falsifying
    the prior:

    1. **Microphone TCC** — probe the responsible app's mic auth status. When
       the daemon is iTerm's grandchild (the typical ``nohup ... &`` setup),
       TCC attributes to iTerm; if iTerm has no mic consent, every roc-send
       through the chain fails silently. This was the actual cause of a
       multi-hour misdiagnosis (mistaken for HAL wedge) on 5/20/2026.
    2. **Log-marker patterns** — scan ``roc_send_log`` for specific libsox
       errors that already tell us the cause: channel-count mismatch is the
       canonical example. Catching these short-circuits past the unreliable
       HAL probe (which uses Claude Remote Mic — itself broken on some Macs
       for reasons unrelated to HAL state).
    3. **HAL wedge** — TCC OK, no log-marker hit, but ``sox -d -n`` (a minimal
       default-input probe) also fails. SoX's CoreAudio backend can't open any
       input — HAL is wedged; empirically only a reboot clears it (``killall
       coreaudiod`` and mass-killing audio clients have been observed NOT to
       work). Earlier versions of this probe used ``roc-send -i core://Claude
       Remote Mic`` and were always false-positive because that virtual device
       is a roc-vad RECEIVER (no upstream data → SoX always fails to open).
    4. **Device-specific** — generic fallthrough when HAL is healthy but the
       requested device couldn't be opened (name collision among same-named
       devices, USB unplugged, exclusive lock, etc.).

    Returns ``(code, title, diagnosis_msg, suggestions)`` for
    ``ResolvableApplyError``.
    """
    # Tier 1: Microphone TCC
    mic_status = await _check_microphone_tcc(service, host)
    if mic_status in ('denied', 'notDetermined'):
        return (
            'microphone-tcc-denied',
            f'Microphone TCC not granted on {host}',
            f'Diagnosis: AVCaptureDevice mic auth status on {host} is `{mic_status}`. '
            'macOS gates microphone access via TCC against the *responsible app* — '
            'when the dispatching daemon is launched as a child of a terminal '
            '(e.g. via `nohup ... &` from iTerm), every roc-send through the chain '
            "inherits the terminal's consent. If that terminal has no Microphone "
            'TCC, the denial is silent and SoX wraps the failure as a generic '
            '"backend dispatcher: failed to open source" error — indistinguishable '
            'from a HAL wedge.',
            (
                f'On {host}: open System Settings → Privacy & Security → Microphone',
                "Enable Microphone access for the daemon's parent app (typically iTerm.app — check `ps` to confirm)",
                f'OR: in that terminal app on {host}, run `sox -d -n trim 0 0.1` '
                'to trigger a TCC prompt, then click Allow',
                'Re-run apply once consent is granted',
            ),
        )
    if mic_status == 'restricted':
        return (
            'microphone-tcc-restricted',
            f'Microphone TCC blocked by policy on {host}',
            f'Diagnosis: AVCaptureDevice mic auth status on {host} is `restricted` '
            '— an MDM profile or parental control is blocking mic access. Cannot '
            'be overridden from a CLI; requires admin intervention.',
            (
                f'Contact the admin managing {host} to grant Microphone access',
                'Or temporarily relax the policy via Profile Manager / Jamf / MDM',
            ),
        )

    # Tier 2: parse roc-send.log for specific libsox markers that already
    # name the cause — short-circuits past the unreliable HAL probe.
    channel_match = re.search(
        r'requested channel count:\s*required_by_input=(\d+)\s+requested_by_user=(\d+)',
        roc_send_log,
    )
    if channel_match:
        device_ch, requested_ch = channel_match.group(1), channel_match.group(2)
        return (
            'roc-send-channel-count-mismatch',
            f'roc-send channel-count mismatch on {host}',
            f'Diagnosis: The input device on {host} has {device_ch} channel(s) but '
            f'roc-send was invoked requesting {requested_ch} channel(s). libsox '
            "won't negotiate down. The orchestrator is supposed to detect the "
            "device's native channel count and pass `-c N` accordingly (task #21) — "
            'this path is currently regressed (the original fix lived in the '
            'ffmpeg pipeline that was reverted in commit 970040d8).',
            (
                f'On {host}: confirm the input device is mono with `claude-coreaudio-volume list`',
                'Workaround: choose an input device whose native channel count matches '
                "roc-send's default (stereo / 2 channels)",
                'Fix: re-engage task #21 — add per-device channel-count detection in '
                '`_roc_send_command` and inject `-c {channels}` into the roc-send args',
            ),
        )

    # Tier 3: HAL probe via `sox -d -n` against the default input device.
    # Earlier versions used `roc-send -i core://Claude Remote Mic` as the
    # probe — but Claude Remote Mic is a roc-vad RECEIVER device. It has no
    # upstream data unless an active RTP feed is wired to it, so SoX trying
    # to open it for read always fails regardless of HAL state. That made
    # the probe a near-guaranteed false-positive "WEDGED" verdict on any
    # host without an active mesh. `sox -d -n trim 0 0.1` exercises SoX's
    # CoreAudio backend against the default input — succeeds if the path
    # is healthy, fails if HAL is genuinely wedged.
    probe = await _run(
        service,
        host,
        'gtimeout 3 sox -d -n trim 0 0.1 2>/tmp/wedge-probe.log >/dev/null && echo HEALTHY || echo WEDGED',
    )
    verdict = probe.stdout.strip().split('\n')[-1]
    if verdict == 'WEDGED':
        return (
            'core-audio-hal-wedge',
            f'CoreAudio HAL wedge on {host}',
            f'Diagnosis: Microphone TCC is `{mic_status}` (good), no libsox log marker '
            'matched the channel-count or other recognized patterns, but `sox -d -n` '
            "(a minimal default-device probe) ALSO fails. SoX's CoreAudio backend "
            f"can't open any input on {host} — the HAL itself is wedged.",
            (
                f'Reboot {host} — empirically, `killall coreaudiod`, killing '
                '`AudioComponentRegistrar`, and mass-killing every audio-framework '
                'client process (including the roc_vad HAL plugin host) have all '
                'been observed to NOT clear the wedge state. Reboot is the '
                'documented and reliable recovery.',
                'After reboot, re-run apply',
            ),
        )

    # Tier 4: Device-specific (no marker matched, HAL probe says healthy)
    return (
        'roc-send-device-specific-open-failure',
        f'roc-send cannot open the requested input device on {host}',
        f'Diagnosis: Microphone TCC is `{mic_status}` (good), CoreAudio HAL is '
        'healthy (the roc-vad virtual mic opens), no recognized libsox marker '
        "matched the log, but this particular device couldn't be opened. "
        'Likely: device unplugged, exclusively held by another process, or '
        'format negotiation failure.',
        (
            'Check that the device is connected to the hub and recognized by Core Audio',
            'Check that no other process holds the device exclusively',
            'Inspect `/tmp/roc-send.log` on the hub for additional clues',
        ),
    )


# -- Peer phase ---------------------------------------------------------------


async def _apply_peer(service: DispatchService, plan: _Plan, peer: str) -> HostApplyOutcome:
    """Ensure peer has Claude Remote Mic + Claude Remote Speaker devices wired to the hub.

    When the hub is managing output (``--output`` was passed), the peer's default
    output is also flipped to ``Claude Remote Speaker`` so peer-app audio enters
    the RTP path. Symmetrically, when the hub is managing input (``--input``),
    the peer's default input is flipped to ``Claude Remote Mic`` so peer-app mic
    reads consume the mesh-broadcast hub mic instead of whatever local device
    happened to be set.

    Tears down any stale roc-send / roc-recv / sox-pipe from when ``peer`` was
    previously the hub — without this, hub-flip transitions leave the old hub's
    mic broadcasting indefinitely. Every peer's Claude Remote Mic would then
    receive N audio streams mixed (one per Mac that has ever been hub), causing
    chipmunk-pitched playback when streams overlap.
    """
    actions: list[str] = []
    actions.extend(await _teardown_stale_hub_processes(service, peer))
    list_text = (await _run(service, peer, 'roc-vad device list')).stdout
    actions.extend(await _ensure_claude_remote_mic(service, peer, list_text))
    actions.extend(await _ensure_peer_speaker(service, plan, peer, list_text))
    if plan.output_device is not None:
        actions.extend(await _route_peer_output_to_hub(service, peer))
    if plan.input_device is not None:
        actions.extend(await _route_peer_input_from_hub(service, peer))
    return HostApplyOutcome(host=peer, role='peer', actions=actions, success=True)


async def _teardown_stale_hub_processes(service: DispatchService, peer: str) -> Sequence[str]:
    """Kill any roc-send / roc-recv / sox-pipe left over from when ``peer`` was last hub.

    Empirically observed 2026-05-22/23: hub-flip transitions (M5→M2→M3→M5)
    left the former hub's roc-send running and broadcasting indefinitely.
    ``_restart_roc_send`` does its own pkill on the NEW hub before relaunching,
    but never reaches demoting Macs through the peer-apply path. This helper
    closes that gap so the post-flip mesh has exactly one mic broadcaster.

    Reports an action only when something was killed — steady-state applies on
    peers that never were hub produce no output line, keeping logs clean.
    """
    result = await _run(
        service,
        peer,
        'killed=0; '
        'for pat in "roc-send" "roc-recv" "sox --buffer 524288 -t coreaudio"; do '
        '  pgrep -fc "$pat" >/dev/null 2>&1 && { pkill -f "$pat" 2>/dev/null; killed=1; }; '
        'done; '
        'echo "killed=$killed"',
    )
    return ['tore down stale hub processes'] if 'killed=1' in result.stdout else []


async def _route_peer_output_to_hub(service: DispatchService, peer: str) -> Sequence[str]:
    """Route ``peer``'s default + system-effects output to Claude Remote Speaker at max volume.

    Volume pinned to max because Claude Remote Speaker is a digital intermediate;
    gain belongs at the hub's analog endpoint, which the user controls.
    """
    await _run(service, peer, 'SwitchAudioSource -t output -s "Claude Remote Speaker"')
    await _run(service, peer, 'SwitchAudioSource -t system -s "Claude Remote Speaker"')
    await _run(service, peer, 'claude-coreaudio-volume set "Claude Remote Speaker" output 1.0')
    return ['set default output + system output → Claude Remote Speaker (volume → max)']


async def _route_peer_input_from_hub(service: DispatchService, peer: str) -> Sequence[str]:
    """Set ``peer``'s Core Audio default input to ``Claude Remote Mic`` at full volume.

    Volume pinned to max because Claude Remote Mic is a digital intermediate;
    gain belongs at the hub's physical mic.
    """
    await _run(service, peer, 'SwitchAudioSource -t input -s "Claude Remote Mic"')
    await _run(service, peer, 'claude-coreaudio-volume set "Claude Remote Mic" input 1.0')
    return ['set default input → Claude Remote Mic (volume → max)']


async def _ensure_claude_remote_mic(service: DispatchService, host: str, list_text: str) -> Sequence[str]:
    """Ensure ``host`` has a Claude Remote Mic receiver bound to ``rtp://0.0.0.0:10001``.

    Role-agnostic — used for both peers (so apps reading the mesh mic find a
    device) and the hub (the hub's roc-send self-loopback ships mic packets to
    127.0.0.1:10001; without this receiver, those packets have no consumer and
    apps on the hub can't read the mesh mic).
    """
    actions: list[str] = []
    mic_idx = _parse_device_index(list_text, name='Claude Remote Mic')
    if mic_idx is None:
        add_out = await _run(service, host, 'roc-vad device add receiver -n "Claude Remote Mic" -r 48000')
        mic_idx = _parse_added_idx(add_out.stdout)
        actions.append(f'added Claude Remote Mic (idx={mic_idx})')

    if mic_idx is None:
        return actions

    mic_show = (await _run(service, host, f'roc-vad device show {mic_idx}')).stdout
    if f'rtp://0.0.0.0:{_MIC_RECEIVE_PORT}' not in mic_show:
        await _run(service, host, f'roc-vad device bind {mic_idx} --source rtp://0.0.0.0:{_MIC_RECEIVE_PORT}')
        actions.append(f'bound Claude Remote Mic → 0.0.0.0:{_MIC_RECEIVE_PORT}')
    return actions


async def _ensure_peer_speaker(
    service: DispatchService,
    plan: _Plan,
    peer: str,
    list_text: str,
) -> Sequence[str]:
    """Ensure peer's Claude Remote Speaker has exactly one slot pointing at the hub.

    ``roc-vad device connect --slot N`` errors against an occupied slot, and there
    is no ``disconnect`` verb — the only way to retire stale slot URIs is to
    ``del`` and re-``add`` the device. Recreate when the live slots diverge from
    ``[target_uri]``; preserve the UID so Core Audio identifies the recreated
    device as the same one reappearing (default-output bindings survive).
    """
    actions: list[str] = []
    target_uri = f'rtp://{plan.hub_ip}:{plan.topology_peer_to_port[peer]}'

    speaker_idx = _parse_device_index(list_text, name='Claude Remote Speaker')
    if speaker_idx is None:
        speaker_idx = await _add_peer_speaker(service, peer, uid=None)
        actions.append(f'added Claude Remote Speaker (idx={speaker_idx})')
        await _connect_speaker_slot_zero(service, peer, speaker_idx, target_uri)
        actions.append(f'connected Claude Remote Speaker slot 0 → {target_uri}')
        return actions

    show_text = (await _run(service, peer, f'roc-vad device show {speaker_idx}')).stdout
    existing_uris = re.findall(r'audiosrc:\s+(rtp://\S+)', show_text)
    if existing_uris == [target_uri]:
        return actions

    uid = _parse_device_uid(show_text)
    await _run(service, peer, f'roc-vad device del {speaker_idx}')
    actions.append(f'deleted Claude Remote Speaker to retire {len(existing_uris)} slot(s): {existing_uris}')
    speaker_idx = await _add_peer_speaker(service, peer, uid=uid)
    await _connect_speaker_slot_zero(service, peer, speaker_idx, target_uri)
    actions.append(f'recreated Claude Remote Speaker (idx={speaker_idx}), slot 0 → {target_uri}')
    return actions


async def _add_peer_speaker(service: DispatchService, host: str, *, uid: str | None) -> int:
    """Create the Claude Remote Speaker sender on ``host``; preserve UID across recreates.

    Pinning the UID lets Core Audio treat the recreated device as the same one
    reappearing — default-output assignments and per-app preferences that
    referenced the old UID stay bound through a del+add cycle.
    """
    uid_arg = f' --uid {uid}' if uid else ''
    cmd = f'roc-vad device add sender -n "Claude Remote Speaker" -r 48000 --fec-encoding disable{uid_arg}'
    add_out = await _run(service, host, cmd)
    idx = _parse_added_idx(add_out.stdout)
    if idx is None:
        raise ApplyError(
            f'{host}: failed to parse Claude Remote Speaker index from `roc-vad device add` output: {add_out.stdout!r}'
        )
    return idx


async def _connect_speaker_slot_zero(service: DispatchService, host: str, speaker_idx: int, target_uri: str) -> None:
    """Wire slot 0 of the Claude Remote Speaker at ``speaker_idx`` to ``target_uri``."""
    await _run(service, host, f'roc-vad device connect {speaker_idx} --slot 0 --source {target_uri}')


def _parse_device_uid(show_text: str) -> str | None:
    """Extract the ``uid:`` field from ``roc-vad device show`` output."""
    match = re.search(r'^\s*uid:\s+(\S+)', show_text, re.MULTILINE)
    return match.group(1) if match else None


# -- roc-vad state readers ----------------------------------------------------


async def _read_existing_peer_port(
    service: DispatchService,
    peer: str,
    hub_ip: str,
) -> int | None:
    """Return the peer's existing Claude Remote Speaker port, preferring current-hub-IP match.

    Pass 1: an ``audiosrc`` line pointing at the current hub IP — that's the active intent.
    Pass 2: any existing ``audiosrc`` line — preserves user-intended port even if hub IP drifted.
    Pass 3: ``None`` — truly fresh peer; caller falls back to alpha assignment.
    """
    list_text = (await _run(service, peer, 'roc-vad device list')).stdout
    speaker_idx = _parse_device_index(list_text, name='Claude Remote Speaker')
    if speaker_idx is None:
        return None
    show_text = (await _run(service, peer, f'roc-vad device show {speaker_idx}')).stdout
    current_hub_match = re.search(
        rf'audiosrc:\s+rtp://{re.escape(hub_ip)}:(\d+)',
        show_text,
    )
    if current_hub_match:
        return int(current_hub_match.group(1))
    any_endpoint_match = re.search(r'audiosrc:\s+rtp://[^:\s]+:(\d+)', show_text)
    if any_endpoint_match:
        return int(any_endpoint_match.group(1))
    return None


async def _read_hub_loopback_port(service: DispatchService, hub_alias: str) -> int | None:
    """Return the port of the hub's own Claude Remote Speaker → 127.0.0.1 endpoint, or ``None``."""
    list_text = (await _run(service, hub_alias, 'roc-vad device list')).stdout
    speaker_idx = _parse_device_index(list_text, name='Claude Remote Speaker')
    if speaker_idx is None:
        return None
    show_text = (await _run(service, hub_alias, f'roc-vad device show {speaker_idx}')).stdout
    match = re.search(r'audiosrc:\s+rtp://127\.0\.0\.1:(\d+)', show_text)
    return int(match.group(1)) if match else None


def _parse_device_index(list_text: str, *, name: str) -> int | None:
    """Find a roc-vad device by exact name in ``device list`` text; return its integer index."""
    for line in list_text.splitlines():
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) < 5:
            continue
        if parts[4].strip() == name:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def _parse_added_idx(stdout: str) -> int | None:
    """Parse ``roc-vad device add`` output for the new device's integer index."""
    match = re.search(r'(?:^|\s)#?(\d+)\b', stdout)
    return int(match.group(1)) if match else None


# -- roc-toolkit command builders ---------------------------------------------


def _roc_recv_command(bind_ports: Sequence[int]) -> str:
    """Shell command that kills any running roc-recv and starts a fresh one bound to the given UDP ports."""
    bind_flags = ' '.join(f'-s rtp://0.0.0.0:{p}' for p in bind_ports)
    return (
        'pkill -f roc-recv 2>/dev/null; sleep 1; '
        f'(nohup /usr/local/bin/roc-recv -o "core://default" {bind_flags} '
        '> /tmp/roc-recv.log 2>&1 < /dev/null &)'
    )


def _roc_send_command(input_device: str, peer_ips: Sequence[str], *, mono: bool) -> str:
    """Shell command that restarts roc-send broadcasting the mic to peer IPs + self-loopback.

    Two paths depending on whether the input device is mono:

    - **Stereo or higher** (``mono=False``): direct ``roc-send -i core://<device>``.
      Real-time HAL scheduling, lowest latency. Multichannel (5.1, 7.1, etc.)
      takes this path — roc-send/SoX accept arbitrary channel counts as long as
      ``>= 2`` (which is what they request by default).
    - **Mono** (``mono=True``): sox captures + upmixes to stereo, pipes raw WAV
      through stdin to ``roc-send -i file:-``. Required because roc-send has no
      channel-count override flag and SoX's CoreAudio backend can't negotiate
      channels down — without the upmix, mono devices die at open. The prior
      ``ffmpeg | roc-send`` pipeline crackled from pipe-buffer underruns
      (default ~64 KB kernel pipe buffer + ffmpeg's default output buffer);
      ``sox --buffer 524288`` enlarges SoX's internal buffer to 512 KB (~2.7s
      of headroom at 48 kHz / 16-bit / 2ch) so the consumer can ride out
      scheduling jitter. WAV (vs raw) carries rate/channels/bit-depth in its
      header so roc-send infers the stream format automatically.
    """
    dest_flags = ' '.join(f'-s rtp://{ip}:{_MIC_RECEIVE_PORT}' for ip in peer_ips)
    dest_flags += f' -s rtp://127.0.0.1:{_MIC_RECEIVE_PORT}'

    if not mono:
        quoted_uri = shlex.quote(f'core://{input_device}')
        return (
            'pkill -f roc-send 2>/dev/null; '
            'pkill -f "sox --buffer 524288 -t coreaudio" 2>/dev/null; '
            'sleep 1; '
            f'(nohup /usr/local/bin/roc-send -i {quoted_uri} {dest_flags} '
            '> /tmp/roc-send.log 2>&1 < /dev/null &)'
        )

    quoted_device = shlex.quote(input_device)
    # ``--input-format=wav`` is mandatory when reading from stdin — roc-send
    # can't sniff the format from the byte stream (``-i file:-``). SoX writes
    # a streaming-WAV header with a placeholder length field (it warns about
    # this); libsox on the read side accepts it fine.
    pipeline = (
        f'sox --buffer 524288 -t coreaudio {quoted_device} '
        '-c 2 -r 48000 -e signed-integer -b 16 -t wav - | '
        f'/usr/local/bin/roc-send --input-format=wav -i file:- {dest_flags}'
    )
    return (
        'pkill -f roc-send 2>/dev/null; '
        'pkill -f "sox --buffer 524288 -t coreaudio" 2>/dev/null; '
        'sleep 1; '
        f'(nohup sh -c {shlex.quote(pipeline)} > /tmp/roc-send.log 2>&1 < /dev/null &)'
    )


# -- Execution + prereq helpers -----------------------------------------------


async def _run(service: DispatchService, host: str, command: str) -> HostRunResult:
    """Execute ``command`` on ``host`` via ``DispatchService``; raise ``ApplyError`` on failure.

    Wraps both dispatch-layer errors (host unreachable, auth, protocol — surfaced in
    ``HostRunResult.error``) and non-zero remote exit codes into a single user-facing
    ``ApplyError``. The orchestrator never sees raw ``RemoteBashError`` exceptions.
    Every dispatched command + its result is traced at DEBUG so the per-run log file
    has a complete cross-machine record.
    """
    logger.debug('%s: dispatch: %s', host, command)
    result = await service.run_target(
        host,
        command,
        session_id=_DISPATCH_SESSION,
        agent_id=None,
        timeout=_DISPATCH_TIMEOUT_S,
    )
    if not result.results:
        raise ApplyError(f'{host}: no host results returned from dispatch')
    host_result = result.results[0]
    logger.debug(
        '%s: result: exit=%s stdout=%r stderr=%r',
        host,
        host_result.exit_code,
        host_result.stdout[-500:],
        host_result.stderr[-500:],
    )
    if host_result.error is not None:
        raise ApplyError(f'{host}: dispatch failed: {host_result.error}')
    if host_result.exit_code != 0:
        raise ApplyError(
            f'{host}: command failed (exit {host_result.exit_code}): {command[:80]}\n'
            f'  stderr: {host_result.stderr.strip() or "<empty>"}'
        )
    return host_result


async def _ensure_prereqs(
    *,
    service: DispatchService,
    hub_alias: str,
    target_peer_aliases: Sequence[str],
    has_output: bool,
    has_input: bool,
    install_prereqs: bool,
) -> None:
    """Make sure hub and target peers have required binaries before any mutation.

    With ``install_prereqs``, hosts that fail the probe get ``scripts/bootstrap.sh``
    dispatched to them (idempotent — brew installs short-circuit when already
    present; roc-toolkit compiles from source on first run, ~5 min). Every host
    that would be installed-into shows a native macOS dialog first: a password
    prompt where sudo is required (the prompt doubles as install consent), or a
    yes/no confirm dialog where ``/usr/local`` is already user-writable. The
    same password entered for one host is silently tried on subsequent hosts
    via ``sudo -K; sudo -S -v`` — when it validates, the user gets a yes/no
    confirm instead of re-typing.
    """
    hub_binaries = ['roc-vad', 'SwitchAudioSource']
    if has_output:
        hub_binaries += ['roc-recv']
    if has_input:
        # claude-tcc-probe powers the Tier-1 Microphone-TCC check in the
        # roc-send-start-failure diagnostic — without it the diagnostic falls
        # through to the HAL probe and risks misclassifying TCC denials as
        # HAL wedges (the misdiagnosis that motivated shipping the probe).
        hub_binaries += ['roc-send', 'claude-coreaudio-volume', 'claude-tcc-probe']

    peer_binaries = ['roc-vad']
    if has_output or has_input:
        # Peers flip their default output/input to Claude Remote Speaker/Mic via
        # SwitchAudioSource so their audio enters/exits the RTP path. The volume
        # tool pins the digital intermediates to max so attenuation only happens
        # at the analog endpoints.
        peer_binaries += ['SwitchAudioSource', 'claude-coreaudio-volume']

    host_reqs: dict[str, Sequence[str]] = {hub_alias: hub_binaries}
    for peer in target_peer_aliases:
        host_reqs[peer] = peer_binaries

    logger.info('prereqs: probing %d host(s)', len(host_reqs))
    missing_by_host = await _probe_missing_binaries(service, host_reqs)

    if any(missing_by_host.values()):
        summary = ', '.join(f'{h}=[{", ".join(m)}]' for h, m in missing_by_host.items() if m)
        logger.info('prereqs missing: %s', summary)
    else:
        logger.info('prereqs satisfied on all hosts')

    if install_prereqs:
        # `--install-prereqs` means "ensure latest binaries deployed," not "only
        # install when something is missing." The bootstrap is idempotent for
        # heavy steps (brew checks skip-if-installed, roc-toolkit checks
        # `command -v roc-send`) but ALWAYS recompiles the Swift CLIs when their
        # source env vars are set — which is what we need to redeploy fixes to
        # claude-tcc-probe / claude-coreaudio-volume without uninstalling first.
        # If nothing is missing, bootstrap every host that has any requirements;
        # if something IS missing, bootstrap only those (saves password dance on
        # hosts whose binaries are fine).
        hosts_to_bootstrap = (
            [h for h, m in missing_by_host.items() if m] if any(missing_by_host.values()) else list(host_reqs.keys())
        )
        logger.info('prereqs: authorizing installs on %s', hosts_to_bootstrap)
        passwords = await _authorize_installs(service, hosts_to_bootstrap)
        logger.info('prereqs: dispatching bootstrap to %s (parallel)', hosts_to_bootstrap)
        await asyncio.gather(*(_install_prereqs_on_host(service, h, passwords[h]) for h in hosts_to_bootstrap))
        logger.info('prereqs: re-probing %s', hosts_to_bootstrap)
        missing_by_host = await _probe_missing_binaries(service, {h: host_reqs[h] for h in hosts_to_bootstrap})
        if not any(missing_by_host.values()):
            logger.info('prereqs: all hosts satisfied after bootstrap')

    if any(missing_by_host.values()):
        issues = [f'{h}: {", ".join(m)}' for h, m in missing_by_host.items() if m]
        hint = (
            '\n\nBootstrap was attempted but binaries are still missing — review the bootstrap output above.'
            if install_prereqs
            else '\n\nRe-run with --install-prereqs to bootstrap these hosts (or see the claude-remote-audio README).'
        )
        body = '\n  '.join(issues)
        raise ApplyError(f'missing prerequisite(s):\n  {body}{hint}')


async def _probe_missing_binaries(
    service: DispatchService,
    host_reqs: Mapping[str, Sequence[str]],
) -> Mapping[str, Sequence[str]]:
    """Return ``host -> missing-binaries`` for each entry in ``host_reqs``.

    Runs ``command -v`` per binary in a single dispatch per host (in parallel
    across hosts), and parses ``MISSING:<name>`` lines back into a list.
    """

    async def _one(host: str, binaries: Sequence[str]) -> tuple[str, Sequence[str]]:
        quoted = [shlex.quote(b) for b in binaries]
        cmd = '; '.join(f'command -v {q} >/dev/null 2>&1 || echo MISSING:{q}' for q in quoted)
        result = await _run(service, host, cmd)
        return host, re.findall(r'MISSING:(\S+)', result.stdout)

    pairs = await asyncio.gather(*(_one(h, bs) for h, bs in host_reqs.items()))
    return dict(pairs)


_KNOWN_PASSWORD_TRY_LIMIT = 2  # most-recent passwords to silently retry per host
_PASSWORD_RETRY_LIMIT = 3  # max prompts per host when the user mistypes


async def _authorize_installs(
    service: DispatchService,
    hosts: Sequence[str],
) -> Mapping[str, str | None]:
    """Walk through ``hosts`` collecting per-host install consent and sudo credentials.

    For each host:

    1. Probe whether ``/usr/local`` is user-writable. If yes, pop a yes/no
       confirm dialog (``Install prereqs on M3 (~5 min)?``); on accept, the
       password is ``None``.
    2. Otherwise, try the most-recently-entered passwords silently (up to
       ``_KNOWN_PASSWORD_TRY_LIMIT``) via ``sudo -K; echo $pw | sudo -S -p "" -v``.
       On a match, pop a yes/no confirm dialog ("using saved password") and
       reuse the validated password.
    3. Otherwise, pop the password dialog. Validate each entry, re-prompt
       on mismatch up to ``_PASSWORD_RETRY_LIMIT`` times.

    Cancel on any dialog raises ``ApplyError`` — no silent skipping.
    """
    known_passwords: list[str] = []
    out: dict[str, str | None] = {}
    for host in hosts:
        password = await _authorize_install_for_host(service, host, known_passwords)
        if password is not None and password not in known_passwords:
            known_passwords.append(password)
        out[host] = password
    return out


async def _authorize_install_for_host(
    service: DispatchService,
    host: str,
    known_passwords: Sequence[str],
) -> str | None:
    """Resolve one host's install consent + sudo password (see ``_authorize_installs``)."""
    if not await _probe_needs_sudo(service, host):
        logger.info('%s: no sudo needed — awaiting install consent', host)
        if not await asyncio.to_thread(_confirm_install_dialog, host, with_saved_password=False):
            raise ApplyError(f'install on {host!r} cancelled by user')
        logger.info('%s: install authorized (no sudo)', host)
        return None

    for pw in reversed(known_passwords[-_KNOWN_PASSWORD_TRY_LIMIT:]):
        if await _validate_sudo_password(service, host, pw):
            logger.info('%s: saved password validated — awaiting install consent', host)
            if not await asyncio.to_thread(_confirm_install_dialog, host, with_saved_password=True):
                raise ApplyError(f'install on {host!r} cancelled by user')
            logger.info('%s: install authorized (saved password)', host)
            return pw

    logger.info('%s: prompting for sudo password', host)
    for attempt in range(_PASSWORD_RETRY_LIMIT):
        password = await asyncio.to_thread(_prompt_sudo_password_dialog, host, attempt > 0)
        if await _validate_sudo_password(service, host, password):
            logger.info('%s: install authorized (fresh password)', host)
            return password
        logger.info('%s: password validation failed (attempt %d/%d)', host, attempt + 1, _PASSWORD_RETRY_LIMIT)
    raise ApplyError(f'sudo password for {host!r} failed validation {_PASSWORD_RETRY_LIMIT} times — aborting install.')


async def _probe_needs_sudo(service: DispatchService, host: str) -> bool:
    """True if scons install on ``host`` would need sudo (any /usr/local subdir not writable)."""
    cmd = '[[ -w /usr/local/bin && -w /usr/local/lib && -w /usr/local/include ]] && echo OK || echo NEED_SUDO'
    result = await _run(service, host, cmd)
    return 'NEED_SUDO' in result.stdout


async def _validate_sudo_password(service: DispatchService, host: str, password: str) -> bool:
    """Silently validate ``password`` against ``host``'s sudo via ``sudo -K`` + ``sudo -S -v``.

    ``sudo -K`` clears any cached credential first so a stale timestamp can't
    masquerade as a successful validation. ``sudo -S -v`` reads from stdin
    and only refreshes the timestamp — no command is executed under privilege.
    """
    quoted = shlex.quote(password)
    cmd = f'sudo -K; echo {quoted} | sudo -S -p "" -v 2>/dev/null'
    result = await service.run_target(
        host,
        cmd,
        session_id=_DISPATCH_SESSION,
        agent_id=None,
        timeout=15.0,
    )
    if not result.results:
        return False
    return result.results[0].exit_code == 0


def _confirm_install_dialog(host: str, *, with_saved_password: bool) -> bool:
    """Pop a yes/no install-consent dialog for ``host``. Returns ``True`` on accept."""
    suffix = ' (using saved password)' if with_saved_password else ''
    # `with timeout` + `giving up after` — see _prompt_sudo_password_dialog for rationale.
    script = (
        'with timeout of 86400 seconds\n'
        'tell application "System Events" to activate\n'
        'tell application "System Events" to '
        f'display dialog "Install prereqs on {host}{suffix} (~5 min)?" '
        'with title "claude-remote-audio install-prereqs" '
        'buttons {"Cancel", "Install"} default button "Install" '
        'giving up after 86400\n'
        'end timeout'
    )
    return subprocess.run(['osascript', '-e', script], capture_output=True, text=True, check=False).returncode == 0


def _prompt_sudo_password_dialog(host: str, retry: bool) -> str:
    """Pop a macOS native password dialog asking for ``host``'s sudo password.

    Uses ``osascript display dialog`` with ``hidden answer`` for a real
    Mac-native input field. Returns the entered text. Raises ``ApplyError``
    when the user cancels. ``retry`` swaps the message to a "wrong password"
    variant for subsequent attempts.
    """
    prompt = f'Wrong password — sudo password for {host}:' if retry else f'sudo password for {host}:'
    # AppleScript has two independent timeouts that BOTH default to short
    # values in non-TTY osascript contexts:
    #   - `giving up after N`     gates the dialog's own auto-dismiss
    #   - `with timeout of N`     gates the AppleEvent `tell` to System Events
    # Without `with timeout`, osascript times out the tell call after seconds,
    # exits non-zero, and the dialog is left orphaned in System Events (still
    # visible, but no listener for the user's response).
    script = (
        'with timeout of 86400 seconds\n'
        'tell application "System Events" to activate\n'
        'tell application "System Events" to '
        f'display dialog "{prompt}" '
        'with title "claude-remote-audio install-prereqs" '
        'default answer "" with hidden answer '
        'buttons {"Cancel", "OK"} default button "OK" '
        'giving up after 86400\n'
        'end timeout'
    )
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ApplyError(f'sudo password prompt for {host!r} cancelled — aborting install.') from exc
    # osascript's `display dialog` returns a record like
    # `button returned:OK, text returned:thepass, gave up:false` when `giving
    # up after` is present (without it, only the first two fields appear).
    # Lookahead stops the capture at `, gave up:` if present, else at EOL —
    # so the password is captured cleanly in both shapes.
    match = re.search(r'text returned:(.*?)(?=, gave up:|$)', result.stdout.rstrip('\n'))
    if match is None:
        raise ApplyError(f'unexpected osascript output: {result.stdout!r}')
    return match.group(1)


async def _install_prereqs_on_host(
    service: DispatchService,
    host: str,
    sudo_password: str | None,
) -> None:
    """Dispatch ``scripts/bootstrap.sh`` to ``host`` and run it.

    The script is base64-encoded so it ships through dispatch as a single
    command (no heredoc quoting, no temp files, no mount). When
    ``sudo_password`` is set, an ``export SUDO_PASSWORD=...`` line is prepended
    to the payload — the script's install step pipes that into ``sudo -S``.
    Swift sources for ``claude-coreaudio-volume`` and ``claude-tcc-probe`` are
    base64-encoded and shipped via env vars (``CRA_SWIFT_CLAUDE_*_B64``);
    bootstrap.sh decodes, compiles with ``swiftc``, and installs each binary.
    Timeout is generous (15 min) because cold first runs include a scons
    compile of roc-toolkit (~5 min on Apple Silicon).
    """
    script_text = _bootstrap_script_path().read_text(encoding='utf-8')
    volume_b64 = base64.b64encode(_coreaudio_volume_swift_source_path().read_bytes()).decode('ascii')
    tcc_probe_b64 = base64.b64encode(_tcc_probe_swift_source_path().read_bytes()).decode('ascii')
    exports = [
        f'export CRA_SWIFT_CLAUDE_COREAUDIO_VOLUME_B64={shlex.quote(volume_b64)}',
        f'export CRA_SWIFT_CLAUDE_TCC_PROBE_B64={shlex.quote(tcc_probe_b64)}',
    ]
    if sudo_password is not None:
        exports.append(f'export SUDO_PASSWORD={shlex.quote(sudo_password)}')
    payload = '\n'.join((*exports, script_text))
    encoded = base64.b64encode(payload.encode('utf-8')).decode('ascii')
    # Write the script to a temp file and exec with stdin from /dev/null.
    # Piping the script to `bash` (the obvious variant) leaves bash's stdin
    # as the pipe — any interactive child (brew install with auto-update
    # prompts, scons, etc.) inherits and consumes the rest of the script as
    # if it were user input, silently swallowing every line past the first
    # interactive command. The file-plus-`</dev/null` form gives child
    # commands a clean empty stdin.
    cmd = f'echo {shlex.quote(encoded)} | base64 -d > /tmp/cra-bootstrap.sh && bash /tmp/cra-bootstrap.sh </dev/null'

    logger.info('%s: bootstrap dispatching (timeout 15min; cold compile ~5min)', host)
    started = asyncio.get_event_loop().time()
    result = await service.run_target(
        host,
        cmd,
        session_id=_DISPATCH_SESSION,
        agent_id=None,
        timeout=900.0,
    )
    elapsed = asyncio.get_event_loop().time() - started
    if not result.results:
        raise ApplyError(f'{host}: bootstrap dispatch returned no host result')
    hr = result.results[0]
    logger.debug('%s: bootstrap stdout:\n%s', host, hr.stdout)
    logger.debug('%s: bootstrap stderr:\n%s', host, hr.stderr)
    if hr.error is not None or hr.exit_code != 0:
        tail = (hr.stderr or hr.stdout or '').strip()[-2000:]
        raise ApplyError(f'{host}: bootstrap failed (exit {hr.exit_code}):\n{tail}')
    logger.info('%s: bootstrap complete in %.1fs', host, elapsed)


def _bootstrap_script_path() -> Path:
    """Locate ``scripts/bootstrap.sh`` next to the ``claude_remote_audio`` package."""
    return Path(__file__).resolve().parent.parent / 'scripts' / 'bootstrap.sh'


def _coreaudio_volume_swift_source_path() -> Path:
    """Locate ``swift/claude-coreaudio-volume.swift`` next to the package."""
    return Path(__file__).resolve().parent.parent / 'swift' / 'claude-coreaudio-volume.swift'


def _tcc_probe_swift_source_path() -> Path:
    """Locate ``swift/claude-tcc-probe.swift`` next to the package."""
    return Path(__file__).resolve().parent.parent / 'swift' / 'claude-tcc-probe.swift'
