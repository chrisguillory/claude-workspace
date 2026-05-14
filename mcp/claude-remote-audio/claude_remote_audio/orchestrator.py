from __future__ import annotations

import asyncio
import difflib
import re
import shlex
import unicodedata
from collections.abc import Mapping, Sequence
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
) -> ApplyResult:
    """Resolve ``target`` into hosts and converge each toward the declared audio topology.

    ``hub=None`` defaults to the local machine — the discovered daemon whose advertised
    IPs overlap with this host's interface IPs (``DiscoveredHost.is_self``). Locality
    model: the command acts on the machine you ran it from. Pass ``hub`` explicitly to
    override.
    """
    service = DispatchService()
    plan = await _build_plan(
        service=service,
        target=target,
        hub=hub,
        input_device=input_device,
        output_device=output_device,
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
) -> _Plan:
    """Parse ``--target`` against mDNS discovery + client groups, resolve hub default if absent."""
    hosts = await browse_hosts(timeout=3.0)
    aliases: dict[str, DiscoveredHost] = {h.alias.lower(): h for h in hosts}
    if not aliases:
        raise ApplyError('no daemons discovered on the LAN — ensure claude-remote-bash-daemon is running on each Mac')

    if hub is None:
        self_host = next((h for h in hosts if h.is_self), None)
        if self_host is None:
            raise ApplyError(
                'no --hub provided and the local machine is not a discovered daemon. '
                f'Discovered: {sorted(aliases.keys())}. '
                'Start claude-remote-bash-daemon locally or pass --hub explicitly.'
            )
        hub = self_host.alias

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

    await _check_prereqs(
        service=service,
        hub_alias=hub,
        target_peer_aliases=target_peer_aliases,
        has_output=output_device is not None,
        has_input=input_device is not None,
    )

    if output_device is not None:
        await _ensure_bluetooth_output(service, hub, output_device)

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
    """Pick a single IPv4 address from a host's mDNS-advertised set (first wins for v0)."""
    if not host.ips:
        raise ApplyError(f'host {host.alias!r} has no advertised IPs')
    return host.ips[0]


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


async def _ensure_bluetooth_output(service: DispatchService, hub_alias: str, output_device: str) -> None:
    """If ``output_device`` is a paired Bluetooth audio device on the hub, bring up the link.

    Called from ``_build_plan`` *before* ``enumerate_devices`` so the freshly-connected
    device appears in Core Audio's output list when we enumerate. No-op when the device
    is wired, AirPlay, virtual, or simply not paired on this hub — the subsequent
    ``_resolve_output_device`` step will surface a clear "not found" error if it doesn't
    appear in Core Audio after this step.

    Cold-state caveat: when the device has *never* been engaged on this hub, ``blueutil
    --connect`` brings up the BT control link but A2DP doesn't auto-engage; Core Audio
    still won't list the device. The remedy in that case is a one-time click in the
    Sound menu on the hub Mac. Subsequent applies work without the click (warm state).
    """
    try:
        bt_device = await bluetooth.find_device(service, hub_alias, output_device)
    except bluetooth.BluetoothError as exc:
        raise ApplyError(f'{hub_alias}: bluetooth probe failed: {exc}') from exc

    if bt_device is None or bt_device.connected:
        return

    try:
        await bluetooth.ensure_connected(service, hub_alias, bt_device.address)
    except bluetooth.BluetoothError as exc:
        raise ApplyError(f'{hub_alias}: failed to connect Bluetooth device {bt_device.name!r}: {exc}') from exc

    # A2DP engagement / Core Audio registration lags the BT control link by ~1-3s
    # depending on whether Core Audio remembers the device. Wait so the subsequent
    # ``enumerate_devices`` call sees the device in its output list.
    await asyncio.sleep(2)


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
    return HostApplyOutcome(host=plan.hub_alias, role='hub', actions=actions, success=True)


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
    """Kill any running roc-send on the hub; relaunch broadcasting to all topology peers + self-loopback."""
    assert plan.input_device is not None  # guaranteed by caller
    peer_ips = [plan.topology_peer_to_ip[p] for p in plan.topology_peer_aliases]
    await _run(service, plan.hub_alias, _roc_send_command(plan.input_device, peer_ips))
    return [f'restarted roc-send (input={plan.input_device})']


# -- Peer phase ---------------------------------------------------------------


async def _apply_peer(service: DispatchService, plan: _Plan, peer: str) -> HostApplyOutcome:
    """Ensure peer has Claude Remote Mic + Claude Remote Speaker devices wired to the hub."""
    list_text = (await _run(service, peer, 'roc-vad device list')).stdout
    actions: list[str] = []
    actions.extend(await _ensure_peer_mic(service, peer, list_text))
    actions.extend(await _ensure_peer_speaker(service, plan, peer, list_text))
    return HostApplyOutcome(host=peer, role='peer', actions=actions, success=True)


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
    """Ensure peer has a Claude Remote Speaker sender connected to the hub at its assigned port."""
    actions: list[str] = []
    speaker_idx = _parse_device_index(list_text, name='Claude Remote Speaker')
    if speaker_idx is None:
        add_out = await _run(
            service,
            peer,
            'roc-vad device add sender -n "Claude Remote Speaker" -r 48000 --fec-encoding disable',
        )
        speaker_idx = _parse_added_idx(add_out.stdout)
        actions.append(f'added Claude Remote Speaker (idx={speaker_idx})')

    if speaker_idx is None:
        return actions

    target_uri = f'rtp://{plan.hub_ip}:{plan.topology_peer_to_port[peer]}'
    speaker_show = (await _run(service, peer, f'roc-vad device show {speaker_idx}')).stdout
    if target_uri in speaker_show:
        return actions

    existing_slots = sorted({int(m.group(1)) for m in re.finditer(r'slot\s+(\d+):', speaker_show)})
    next_slot = (max(existing_slots) + 1) if existing_slots else 0
    await _run(
        service,
        peer,
        f'roc-vad device connect {speaker_idx} --slot {next_slot} --source {target_uri}',
    )
    actions.append(f'connected Claude Remote Speaker slot {next_slot} → {target_uri}')
    return actions


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
    """Shell command that restarts roc-send broadcasting the mic to peer IPs + self-loopback."""
    quoted_input = shlex.quote(f'core://{input_device}')
    dest_flags = ' '.join(f'-s rtp://{ip}:{_MIC_RECEIVE_PORT}' for ip in peer_ips)
    return (
        'pkill -f roc-send 2>/dev/null; sleep 1; '
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
    """
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
    if host_result.error is not None:
        raise ApplyError(f'{host}: dispatch failed: {host_result.error}')
    if host_result.exit_code != 0:
        raise ApplyError(
            f'{host}: command failed (exit {host_result.exit_code}): {command[:80]}\n'
            f'  stderr: {host_result.stderr.strip() or "<empty>"}'
        )
    return host_result


async def _check_prereqs(
    *,
    service: DispatchService,
    hub_alias: str,
    target_peer_aliases: Sequence[str],
    has_output: bool,
    has_input: bool,
) -> None:
    """Verify hub and target peers have required binaries before any mutation."""
    hub_binaries = ['roc-vad', 'SwitchAudioSource']
    if has_output:
        hub_binaries += ['roc-recv']
    if has_input:
        hub_binaries += ['roc-send']
    await _verify_binaries(service, hub_alias, hub_binaries)

    await asyncio.gather(*(_verify_binaries(service, peer, ['roc-vad']) for peer in target_peer_aliases))


async def _verify_binaries(service: DispatchService, host: str, binaries: Sequence[str]) -> None:
    """``command -v`` each binary on ``host``; raise ``ApplyError`` listing any missing."""
    quoted = [shlex.quote(b) for b in binaries]
    cmd = '; '.join(f'command -v {q} >/dev/null 2>&1 || echo MISSING:{q}' for q in quoted)
    result = await _run(service, host, cmd)
    missing = re.findall(r'MISSING:(\S+)', result.stdout)
    if missing:
        raise ApplyError(
            f'{host}: missing prerequisite(s): {", ".join(missing)}. '
            'See the claude-remote-audio README for install instructions.'
        )
