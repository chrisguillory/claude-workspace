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

__all__ = [
    'ApplyError',
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


class ApplyError(RuntimeError):
    """Configuration error or constraint violation that prevents apply from running."""


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


# -- Hub phase ----------------------------------------------------------------


async def _apply_hub(service: DispatchService, plan: _Plan) -> HostApplyOutcome:
    """Run hub-side mutations: feedback-loop guard, then opt-in output and input management."""
    await _guard_feedback_loop(service, plan)
    actions: list[str] = []
    if plan.output_device is not None:
        actions.extend(await _set_hub_output(service, plan))
        actions.extend(await _restart_roc_recv(service, plan))
    if plan.input_device is not None:
        actions.extend(await _restart_roc_send(service, plan))
        actions.extend(await _set_hub_input_to_remote_mic(service, plan))
    return HostApplyOutcome(host=plan.hub_alias, role='hub', actions=actions, success=True)


async def _set_hub_input_to_remote_mic(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Set the hub's Core Audio default input to ``Claude Remote Mic`` at full volume.

    Apps on the hub then consume the mesh-distributed mic via the self-loopback
    leg of roc-send (127.0.0.1:10001 → Claude Remote Mic), giving consistent
    behavior across the entire mesh: every Mac's "default mic" is the hub's
    physical input. Volume is pinned to max for the same reason as on peers —
    Claude Remote Mic is a digital intermediate; gain belongs at the physical
    mic. Crucially this only touches the INPUT side; the hub's OUTPUT (which
    is the user's physical headphones, e.g. AirPods) is never touched.
    Tracked for upgrade to a device-by-UID approach in #39.
    """
    await _run(service, plan.hub_alias, 'SwitchAudioSource -t input -s "Claude Remote Mic"')
    await _run(service, plan.hub_alias, "osascript -e 'set volume input volume 100'")
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
    """Resolve user-typed device name to canonical (against pre-fetched list), then set hub default output."""
    assert plan.output_device is not None  # guaranteed by caller
    canonical = _resolve_output_device(plan.hub_outputs, plan.hub_alias, plan.output_device)
    await _run(
        service,
        plan.hub_alias,
        f'SwitchAudioSource -t output -s {shlex.quote(canonical)}',
    )
    return [f'set default output → {canonical}']


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
    """Kill any running roc-recv on the hub; relaunch bound to all topology peer ports + self-loopback."""
    bind_ports = [plan.topology_peer_to_port[p] for p in plan.topology_peer_aliases]
    if plan.hub_loopback_port is not None:
        bind_ports.append(plan.hub_loopback_port)
    await _run(service, plan.hub_alias, _roc_recv_command(bind_ports))
    return ['restarted roc-recv']


async def _restart_roc_send(service: DispatchService, plan: _Plan) -> Sequence[str]:
    """Kill any running roc-send on the hub; relaunch the ffmpeg→roc-send pipeline.

    The pipeline is detached via ``nohup`` so the dispatch returns immediately
    even when the child crashes on startup (bad device name, format mismatch,
    TCC denial, etc.). Verifies survival via a post-launch pgrep so apply fails
    loudly instead of silently reporting success against a dead pipeline.
    """
    assert plan.input_device is not None  # guaranteed by caller
    peer_ips = [plan.topology_peer_to_ip[p] for p in plan.topology_peer_aliases]
    await _run(service, plan.hub_alias, _roc_send_command(plan.input_device, peer_ips))
    await asyncio.sleep(3)
    alive = await _run(service, plan.hub_alias, 'pgrep -f roc-send >/dev/null && echo alive || echo dead')
    if alive.stdout.strip() != 'alive':
        logs = await _run(
            service,
            plan.hub_alias,
            'echo "--- roc-send.log ---"; tail -20 /tmp/roc-send.log 2>/dev/null; '
            'echo "--- ffmpeg-mic.log ---"; tail -20 /tmp/ffmpeg-mic.log 2>/dev/null',
        )
        raise ApplyError(
            f'{plan.hub_alias}: roc-send pipeline died within 3s of launch — '
            f'input={plan.input_device!r}. Diagnostic logs:\n{logs.stdout}'
        )
    return [f'restarted roc-send (input={plan.input_device})']


# -- Peer phase ---------------------------------------------------------------


async def _apply_peer(service: DispatchService, plan: _Plan, peer: str) -> HostApplyOutcome:
    """Ensure peer has Claude Remote Mic + Claude Remote Speaker devices wired to the hub.

    When the hub is managing output (``--output`` was passed), the peer's default
    output is also flipped to ``Claude Remote Speaker`` so peer-app audio enters
    the RTP path. Symmetrically, when the hub is managing input (``--input``),
    the peer's default input is flipped to ``Claude Remote Mic`` so peer-app mic
    reads consume the mesh-broadcast hub mic instead of whatever local device
    happened to be set.
    """
    list_text = (await _run(service, peer, 'roc-vad device list')).stdout
    actions: list[str] = []
    actions.extend(await _ensure_peer_mic(service, peer, list_text))
    actions.extend(await _ensure_peer_speaker(service, plan, peer, list_text))
    if plan.output_device is not None:
        actions.extend(await _route_peer_output_to_hub(service, peer))
    if plan.input_device is not None:
        actions.extend(await _route_peer_input_from_hub(service, peer))
    return HostApplyOutcome(host=peer, role='peer', actions=actions, success=True)


async def _route_peer_output_to_hub(service: DispatchService, peer: str) -> Sequence[str]:
    """Set ``peer``'s Core Audio default output to ``Claude Remote Speaker`` at full volume.

    Volume is pinned to max because Claude Remote Speaker is a digital
    intermediate in the mesh — attenuation belongs at the analog endpoint
    (the hub's physical output, e.g. AirPods), which the user controls.
    The ``osascript set volume output volume 100`` call acts on the current
    default output; the preceding ``SwitchAudioSource`` ensures that's CRS
    on this peer. Tracked for upgrade to a device-by-UID approach in #39.
    """
    await _run(service, peer, 'SwitchAudioSource -t output -s "Claude Remote Speaker"')
    await _run(service, peer, "osascript -e 'set volume output volume 100'")
    return ['set default output → Claude Remote Speaker (volume → max)']


async def _route_peer_input_from_hub(service: DispatchService, peer: str) -> Sequence[str]:
    """Set ``peer``'s Core Audio default input to ``Claude Remote Mic`` at full volume.

    Volume is pinned to max because Claude Remote Mic is a digital intermediate
    in the mesh — gain belongs at the analog endpoint (the hub's physical mic,
    controlled by its hardware knob or hub-side Sound prefs). Tracked for
    upgrade to a device-by-UID approach in #39.
    """
    await _run(service, peer, 'SwitchAudioSource -t input -s "Claude Remote Mic"')
    await _run(service, peer, "osascript -e 'set volume input volume 100'")
    return ['set default input → Claude Remote Mic (volume → max)']


async def _ensure_peer_mic(service: DispatchService, peer: str, list_text: str) -> Sequence[str]:
    """Ensure peer has a Claude Remote Mic receiver bound to ``rtp://0.0.0.0:10001``."""
    actions: list[str] = []
    mic_idx = _parse_device_index(list_text, name='Claude Remote Mic')
    if mic_idx is None:
        add_out = await _run(service, peer, 'roc-vad device add receiver -n "Claude Remote Mic" -r 48000')
        mic_idx = _parse_added_idx(add_out.stdout)
        actions.append(f'added Claude Remote Mic (idx={mic_idx})')

    if mic_idx is None:
        return actions

    mic_show = (await _run(service, peer, f'roc-vad device show {mic_idx}')).stdout
    if f'rtp://0.0.0.0:{_MIC_RECEIVE_PORT}' not in mic_show:
        await _run(service, peer, f'roc-vad device bind {mic_idx} --source rtp://0.0.0.0:{_MIC_RECEIVE_PORT}')
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


def _roc_send_command(input_device: str, peer_ips: Sequence[str]) -> str:
    """Shell command that restarts roc-send broadcasting the mic to peer IPs + self-loopback.

    Direct ``core://<device>`` — roc-send reads CoreAudio with real-time HAL
    scheduling. The earlier ffmpeg→roc-send pipe variant introduced crackling
    from pipe-buffer underruns and was reverted; if SoX fails to open the
    device, the symptom is typically a wedged ``coreaudiod`` on the hub
    (kill it with ``sudo killall coreaudiod`` and it respawns clean).
    """
    quoted_input = shlex.quote(f'core://{input_device}')
    dest_flags = ' '.join(f'-s rtp://{ip}:{_MIC_RECEIVE_PORT}' for ip in peer_ips)
    return (
        'pkill -f roc-send 2>/dev/null; pkill -f "ffmpeg.*-f avfoundation" 2>/dev/null; '
        'sleep 1; '
        f'(nohup /usr/local/bin/roc-send -i {quoted_input} '
        f'{dest_flags} -s rtp://127.0.0.1:{_MIC_RECEIVE_PORT} '
        '> /tmp/roc-send.log 2>&1 < /dev/null &)'
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
        hub_binaries += ['roc-send']

    peer_binaries = ['roc-vad']
    if has_output:
        # Peers' default output flips to Claude Remote Speaker via SwitchAudioSource
        # so their audio enters the RTP path back to the hub.
        peer_binaries += ['SwitchAudioSource']

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

    if install_prereqs and any(missing_by_host.values()):
        hosts_to_bootstrap = [h for h, m in missing_by_host.items() if m]
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
    script = (
        'tell application "System Events" to activate\n'
        'tell application "System Events" to '
        f'display dialog "Install prereqs on {host}{suffix} (~5 min)?" '
        'with title "claude-remote-audio install-prereqs" '
        'buttons {"Cancel", "Install"} default button "Install"'
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
    script = (
        'tell application "System Events" to activate\n'
        'tell application "System Events" to '
        f'display dialog "{prompt}" '
        'with title "claude-remote-audio install-prereqs" '
        'default answer "" with hidden answer '
        'buttons {"Cancel", "OK"} default button "OK"'
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
    match = re.search(r'text returned:(.*)$', result.stdout.rstrip('\n'))
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
    Timeout is generous (15 min) because cold first runs include a scons
    compile of roc-toolkit (~5 min on Apple Silicon).
    """
    script_text = _bootstrap_script_path().read_text(encoding='utf-8')
    payload = (
        f'export SUDO_PASSWORD={shlex.quote(sudo_password)}\n{script_text}'
        if sudo_password is not None
        else script_text
    )
    encoded = base64.b64encode(payload.encode('utf-8')).decode('ascii')
    cmd = f'echo {shlex.quote(encoded)} | base64 -d | bash'

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
