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

__all__ = [
    'ApplyError',
    'ApplyResult',
    'HostApplyOutcome',
    'apply',
]


_BASE_PORT = 11001
_MIC_RECEIVE_PORT = 10001
_DISPATCH_SESSION = 'claude-remote-audio'
_DISPATCH_TIMEOUT_S = 60.0


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
    hub: str,
    input_device: str | None = None,
    output_device: str | None = None,
) -> ApplyResult:
    """Resolve ``target`` into hosts and converge each toward the declared audio topology."""
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
    hub: str,
    input_device: str | None,
    output_device: str | None,
) -> _Plan:
    """Parse ``--target`` against mDNS discovery + client groups, validate against ``--hub``."""
    hosts = await browse_hosts(timeout=3.0)
    aliases: dict[str, DiscoveredHost] = {h.alias.lower(): h for h in hosts}
    if not aliases:
        raise ApplyError('no daemons discovered on the LAN — ensure claude-remote-bash-daemon is running on each Mac')

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

    hub_loopback_port = await _read_hub_loopback_port(service, hub)
    topology_peer_aliases = sorted(a for a in aliases if a != hub_lower)
    topology_peer_to_ip = {p: _best_ip(aliases[p]) for p in topology_peer_aliases}
    topology_peer_to_port = await _assign_peer_ports(service, topology_peer_aliases, hub_ip)

    return _Plan(
        hub_alias=hub,
        hub_in_target=hub_in_target,
        hub_ip=hub_ip,
        hub_loopback_port=hub_loopback_port,
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
    """Resolve user-typed device name to canonical, then set the hub's default output."""
    assert plan.output_device is not None  # guaranteed by caller
    canonical = await _resolve_output_device(service, plan.hub_alias, plan.output_device)
    await _run(
        service,
        plan.hub_alias,
        f'SwitchAudioSource -t output -s {shlex.quote(canonical)}',
    )
    return [f'set default output → {canonical}']


async def _resolve_output_device(service: DispatchService, hub_alias: str, requested: str) -> str:
    """Match ``requested`` against canonical output device names on the hub.

    Strategy (enumerate-then-resolve, not normalize-then-compare):

    1. Exact match against ``SwitchAudioSource -a -t output`` output.
    2. NFKC case-folded match (handles legitimate Unicode equivalences like full-width).
    3. Miss → raise ``ApplyError`` with a ``difflib.get_close_matches`` "did you mean" hint.

    Smart-quote variants (U+2019 vs U+0027) are intentionally NOT silently folded —
    ``SwitchAudioSource -s NAME`` is destructive, so we ask the user to pick the canonical
    name rather than guess. The error message surfaces the canonical name for copy-paste.
    """
    raw = (await _run(service, hub_alias, 'SwitchAudioSource -a -t output')).stdout
    canonical = [line.strip() for line in raw.splitlines() if line.strip()]

    if requested in canonical:
        return requested

    folded_req = _nfkc_casefold(requested)
    matches = [c for c in canonical if _nfkc_casefold(c) == folded_req]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ApplyError(f'{hub_alias}: --output {requested!r} ambiguous: {matches}')

    suggestion = difflib.get_close_matches(requested, canonical, n=1, cutoff=0.6)
    hint = f' Did you mean {suggestion[0]!r}?' if suggestion else ''
    raise ApplyError(f'{hub_alias}: --output {requested!r} not found. Available: {canonical}.{hint}')


def _nfkc_casefold(s: str) -> str:
    """NFKC-normalize and case-fold a string for matching legitimate Unicode equivalences."""
    return unicodedata.normalize('NFKC', s).casefold()


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
    hub_binaries = ['roc-vad']
    if has_output:
        hub_binaries += ['SwitchAudioSource', 'roc-recv']
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
