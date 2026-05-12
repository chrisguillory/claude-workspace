"""Command-line interface for claude-remote-bash."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence, Set
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.utils import get_claude_workspace_config_home_dir
from cc_lib.utils.atomic_write import atomic_write

from claude_remote_bash.auth import load_config
from claude_remote_bash.client_config import load_groups
from claude_remote_bash.discovery import DiscoveredHost, browse_hosts
from claude_remote_bash.exceptions import (
    AuthError,
    DaemonError,
    HostNotFoundError,
    HostUnreachableError,
    RemoteBashError,
)
from claude_remote_bash.models import (
    AuthFail,
    AuthRequest,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
)
from claude_remote_bash.protocol import read_message, write_message
from claude_remote_bash.selector import SelectorError
from claude_remote_bash.selector import parse as parse_selector

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

app = create_app(help='Cross-machine shell execution for Claude Code.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)

CACHE_DIR = get_claude_workspace_config_home_dir() / 'mcp' / 'claude-remote-bash'
CACHE_FILE = CACHE_DIR / 'hosts-cache.json'
CACHE_TTL_SECONDS = 30.0


def main() -> None:
    """Entry point for claude-remote-bash CLI."""
    run_app(app)


# -- Commands ------------------------------------------------------------------


@app.command()
@error_boundary
def execute(
    command: Annotated[str | None, typer.Argument(help='Command to execute (reads stdin if omitted)')] = None,
    target: Annotated[
        str,
        typer.Option('--target', '-t', help='Target selector — host alias, comma-list, or group name'),
    ] = '',
    session_id: Annotated[
        str, typer.Option('--session-id', envvar='CLAUDE_CODE_SESSION_ID', help='Session ID for CWD tracking')
    ] = 'default',
    agent_id: Annotated[
        str | None, typer.Option('--agent-id', envvar='CLAUDE_CODE_AGENT_ID', help='Agent ID for sub-agent isolation')
    ] = None,
    timeout: Annotated[float, typer.Option('--timeout', help='Command timeout in seconds')] = 120.0,
) -> None:
    """Execute a command on one or more remote hosts.

    \b
    The target selector accepts:
      - a host alias (e.g. M2)
      - a comma-separated list (M2,M3,M4)
      - a named group from client_config.json (e.g. fleet)
      - a literal ip:port (192.168.4.22:51648)
      - any mix of the above; whitespace per-atom is stripped, duplicates rejected.
    """
    if not target:
        raise RemoteBashError('--target is required')

    cmd = command or _read_stdin()
    if not cmd:
        raise RemoteBashError('No command provided (pass as argument or pipe via stdin)')

    asyncio.run(_run_target(target, cmd, session_id, agent_id, timeout))


@app.command()
@error_boundary
def discover() -> None:
    """Browse the LAN for claude-remote-bash daemons."""
    hosts = asyncio.run(browse_hosts(timeout=3.0))
    _write_cache(hosts)

    if not hosts:
        typer.echo('No daemons found on the network.')
        typer.echo('')
        typer.echo('If a daemon should be visible:')
        typer.echo('  - Verify the daemon is running on the target:')
        typer.echo('      `pgrep -f claude-remote-bash-daemon`')
        typer.echo('  - Verify network reachability: `ping <target>.local`')
        typer.echo("  - Ensure client and target are on the same LAN segment (mDNS doesn't cross subnets).")
        return

    typer.echo(f'Found {len(hosts)} daemon(s):\n')
    for h in hosts:
        ips_str = ','.join(h.ips) if h.ips else '?'
        typer.echo(f'  {h.alias:<12} {ips_str}:{h.port}  ({h.hostname})  v{h.version}')


# -- Async internals -----------------------------------------------------------


async def _execute_remote_at(
    *,
    ips: Sequence[str],
    port: int,
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
) -> ExecuteResult:
    """Connect to a daemon at a pre-resolved ``(ips, port)``, authenticate, execute.

    Does no discovery I/O — that's the caller's job, so multi-host dispatch
    can call this concurrently from ``asyncio.gather`` without per-host
    cache races.
    """
    reader, writer = await _open_connection_any(ips, port)
    try:
        await _authenticate(reader, writer)

        msg = ExecuteRequest(
            id='cli-1',
            command=command,
            session_id=session_id,
            agent_id=agent_id,
            timeout=timeout,
        )
        await write_message(writer, msg)
        response = await read_message(reader)

        if isinstance(response, ErrorResponse):
            raise DaemonError(response.message)

        if not isinstance(response, ExecuteResult):
            raise DaemonError(f'unexpected response type: {response.type}')

        return response
    finally:
        writer.close()
        await writer.wait_closed()


async def _run_target(
    target: str,
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
) -> None:
    """Resolve the selector, dispatch to one or more hosts, print results, exit.

    Single-host case: stream raw stdout/stderr; exit with the host's exit code.
    Multi-host case: parallel via ``asyncio.gather``; each host's stdout/stderr
    lines prefixed with ``[hostname] ``; summary table at end; exit 0 iff every
    host succeeded.
    """
    groups = load_groups()
    discovered = await _browse_and_cache()
    discovered_aliases: Set[str] = frozenset(str(e.get('alias', '')).lower() for e in discovered)

    try:
        atoms = parse_selector(target, groups=groups, discovered_aliases=discovered_aliases)
    except SelectorError as exc:
        raise RemoteBashError(str(exc)) from exc

    resolved: list[tuple[str, tuple[Sequence[str], int]]] = []
    missing: list[str] = []
    for atom in atoms:
        addr = _lookup_alias(discovered, atom)
        if addr is None:
            missing.append(atom)
        else:
            resolved.append((atom, addr))

    if missing:
        # Cache may be fresh-by-TTL but stale-by-content — a daemon may have
        # just come up. Force one fresh browse and retry the missing atoms.
        discovered = await _browse_fresh_and_cache()
        for atom in missing:
            addr = _lookup_alias(discovered, atom)
            if addr is None:
                _raise_host_not_found(atom)
                raise AssertionError  # unreachable
            resolved.append((atom, addr))

    if len(resolved) == 1:
        atom, (ips, port) = resolved[0]
        result = await _execute_remote_at(
            ips=ips,
            port=port,
            command=command,
            session_id=session_id,
            agent_id=agent_id,
            timeout=timeout,
        )
        if result.stdout:
            typer.echo(result.stdout)
        if result.stderr:
            typer.echo(result.stderr, err=True)
        raise SystemExit(result.exit_code)

    results = await _multi_dispatch(resolved, command, session_id, agent_id, timeout)
    _print_summary(results)
    overall = 0 if all(r.exit_code == 0 and r.error is None for r in results) else 1
    raise SystemExit(overall)


@dataclasses.dataclass
class _HostRunResult:
    atom: str
    exit_code: int
    duration_s: float
    stdout: str
    stderr: str
    error: str | None  # None on success; non-None on connection/auth/protocol failure


async def _multi_dispatch(
    resolved: Sequence[tuple[str, tuple[Sequence[str], int]]],
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
) -> Sequence[_HostRunResult]:
    """Run the command on every (atom, (ips, port)) in parallel via asyncio.gather.

    Each task is wrapped in a per-host timing + exception capture so a single
    bad daemon doesn't bring down the whole batch.
    """

    async def _one(atom: str, ips: Sequence[str], port: int) -> _HostRunResult:
        started = time.monotonic()
        try:
            result = await _execute_remote_at(
                ips=ips,
                port=port,
                command=command,
                session_id=session_id,
                agent_id=agent_id,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 # exception_safety_linter.py: swallowed-exception — multi-host UX: one bad daemon (connection/auth/protocol/timeout) must not abort the batch; per-host failure is captured into the summary table
            return _HostRunResult(
                atom=atom,
                exit_code=-1,
                duration_s=time.monotonic() - started,
                stdout='',
                stderr='',
                error=f'{type(exc).__name__}: {exc}',
            )
        return _HostRunResult(
            atom=atom,
            exit_code=result.exit_code,
            duration_s=time.monotonic() - started,
            stdout=result.stdout,
            stderr=result.stderr,
            error=None,
        )

    results = await asyncio.gather(*(_one(atom, ips, port) for atom, (ips, port) in resolved))

    for r in results:
        for line in r.stdout.splitlines():
            typer.echo(f'[{r.atom}] {line}')
        for line in r.stderr.splitlines():
            typer.echo(f'[{r.atom}] {line}', err=True)

    return results


def _print_summary(results: Sequence[_HostRunResult]) -> None:
    """Print a per-host summary table after multi-host dispatch."""
    width = max(len('host'), *(len(r.atom) for r in results))
    typer.echo('')
    typer.echo(f'{"host":<{width}}  status  exit  duration')
    for r in results:
        status = 'ok' if r.error is None and r.exit_code == 0 else 'failed'
        suffix = f'  ({r.error})' if r.error else ''
        typer.echo(f'{r.atom:<{width}}  {status:<6}  {r.exit_code:<4}  {r.duration_s:.2f}s{suffix}')


CONNECT_TIMEOUT_SECONDS = 2.0


async def _open_connection_any(ips: Sequence[str], port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Try each IP in order with a short per-attempt timeout; return the first success.

    Daemons advertise all their IPv4 addresses via mDNS (LAN + VPN + ethernet).
    We attempt each with a short timeout so an unreachable VPN address fails
    fast and we fall through to the LAN address.
    """
    errors: list[tuple[str, Exception]] = []
    for ip in ips:
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except (TimeoutError, OSError) as exc:
            errors.append((ip, exc))

    attempt_lines = '\n  '.join(f'{ip}:{port} ({type(exc).__name__}: {exc})' for ip, exc in errors)
    hint = _reachability_hint(errors)
    raise HostUnreachableError(f'Could not connect to any advertised address:\n  {attempt_lines}{hint}')


def _reachability_hint(errors: Sequence[tuple[str, Exception]]) -> str:
    """Suggest a likely cause based on the failure signatures of the attempts.

    TimeoutError → packets never reached a listener at that ip:port. Typically
        a stale cache (daemon restarted on a different port), target on a
        different network, or packet filtering between client and target.
    ConnectionRefusedError → TCP reached the host but the port was closed;
        the daemon is not running (or crashed). The macOS Application
        Firewall, notably, does *not* produce this signature — it lets TCP
        complete and then silently blocks the process from reading data,
        which surfaces as an auth-read hang (see _authenticate).
    """
    if errors and all(isinstance(exc, TimeoutError) for _, exc in errors):
        return (
            '\n\nAll attempts timed out. Likely causes:\n'
            '  - Stale cache: daemon may have restarted on a different port.\n'
            '    Run `claude-remote-bash discover` to refresh.\n'
            '  - Target machine is offline or on a different network.\n'
            '  - Packet filtering between client and target.'
        )
    if errors and all(isinstance(exc, ConnectionRefusedError) for _, exc in errors):
        return (
            "\n\nEvery address actively refused the connection — the daemon isn't\n"
            'running on that port. On the target machine, check:\n'
            '  `pgrep -f claude-remote-bash-daemon`'
        )
    return ''


AUTH_TIMEOUT_SECONDS = 5.0


async def _authenticate(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Perform PSK authentication.

    TCP connect succeeding but auth hanging is the signature of the macOS
    Application Firewall in its default state: the kernel completes the TCP
    handshake, but the daemon process never receives the auth payload
    because the firewall has not yet been granted permission for the
    python3.13 binary. We time out the read so that state surfaces as an
    actionable error instead of an indefinite hang.
    """
    cfg = load_config()
    if cfg is None or not cfg.auth_key:
        raise AuthError('No auth key configured. Run: claude-remote-bash-daemon init')

    await write_message(writer, AuthRequest(key=cfg.auth_key))

    try:
        response = await asyncio.wait_for(read_message(reader), timeout=AUTH_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise AuthError(
            f'TCP connected but the daemon did not respond to auth within {AUTH_TIMEOUT_SECONDS:.0f}s.\n'
            '\n'
            'This is the signature of a pending macOS Application Firewall prompt\n'
            'on the target: the kernel accepts the TCP handshake, but the firewall\n'
            'silently blocks the daemon process from receiving the data.\n'
            '\n'
            'On the target machine:\n'
            '  - Look for an "Allow python3.13 to accept incoming network\n'
            '    connections" dialog (may be hidden behind other windows) and\n'
            '    click Allow, OR\n'
            '  - Approve via terminal (no dialog needed):\n'
            '      claude-remote-bash-daemon allow-firewall\n'
            '  - Verify on the target: does the daemon log show\n'
            '    "Connection from (ip, port)"? If yes the firewall is not the\n'
            "    cause; if no, it's confirmed."
        ) from exc

    if isinstance(response, AuthFail):
        raise AuthError(f'Authentication failed: {response.reason}')


async def _browse_fresh_and_cache() -> Sequence[Mapping[str, object]]:
    """Always browse mDNS; write the result to the cache atomically."""
    hosts = await browse_hosts(timeout=3.0)
    _write_cache(hosts)
    return [{'alias': h.alias, 'hostname': h.hostname, 'ips': list(h.ips), 'port': h.port} for h in hosts]


async def _browse_and_cache() -> Sequence[Mapping[str, object]]:
    """Return the discovered-hosts map — cached if fresh, else fresh browse.

    Splits the I/O (cache read + optional browse + cache write) away from
    per-atom lookup (``_lookup_alias``) so the multi-host dispatch path
    can call this once upfront and run a pure-function lookup per atom
    concurrently inside ``asyncio.gather``.
    """
    cached = _read_cache()
    if cached is not None:
        return cached
    return await _browse_fresh_and_cache()


def _lookup_alias(
    discovered: Sequence[Mapping[str, object]],
    atom: str,
) -> tuple[Sequence[str], int] | None:
    """Resolve one atom to (ips, port), or ``None`` on miss. Pure — no I/O.

    The atom is either a literal ``ip:port`` (direct address; bypasses
    discovery) or a host alias to look up in the already-discovered map.
    Hostname-substring matching is intentionally dropped from the old
    ``_resolve_host``: single-user environment with operator-assigned
    aliases, exact match is enough.

    Returns ``None`` on miss so the caller can decide whether to retry
    with a fresh browse before raising — the cache may be fresh-by-TTL
    but stale-by-content (e.g. a new daemon appeared since the last
    browse).
    """
    if ':' in atom:
        parts = atom.rsplit(':', 1)
        return [parts[0]], int(parts[1])

    atom_lower = atom.lower()
    for entry in discovered:
        if str(entry.get('alias', '')).lower() == atom_lower:
            return _cache_ips(entry), int(str(entry['port']))

    return None


def _raise_host_not_found(atom: str) -> None:
    raise HostNotFoundError(
        f'Host not found: {atom}\n'
        'Run `claude-remote-bash discover` to see available hosts.\n'
        '\n'
        'Common causes:\n'
        "  - The daemon isn't running on the target.\n"
        '  - Alias typo — check the output of `discover`.\n'
        "  - Target is on a different LAN segment (mDNS doesn't cross subnets)."
    )


async def _resolve_host(host: str) -> tuple[Sequence[str], int]:
    """Resolve a single host: cache → fresh-browse → raise.

    Used by today's single-host ``execute`` path. The multi-host dispatch
    (later commit) uses ``_browse_and_cache`` + ``_lookup_alias`` directly
    so it can fan out across atoms inside ``asyncio.gather``.
    """
    discovered = await _browse_and_cache()
    result = _lookup_alias(discovered, host)
    if result is None:
        # Cache may be fresh-by-TTL but stale-by-content; force a fresh
        # browse before declaring the host missing.
        discovered = await _browse_fresh_and_cache()
        result = _lookup_alias(discovered, host)
    if result is None:
        _raise_host_not_found(host)
        raise AssertionError  # unreachable — _raise_host_not_found always raises
    return result


def _cache_ips(entry: Mapping[str, object]) -> Sequence[str]:
    """Extract IP list from a cache entry, accepting both new and legacy formats."""
    ips = entry.get('ips')
    if isinstance(ips, list) and ips:
        return [str(ip) for ip in ips]
    legacy = entry.get('ip')
    if isinstance(legacy, str) and legacy:
        return [legacy]
    return []


# -- Helpers -------------------------------------------------------------------


def _read_stdin() -> str:
    """Read command from stdin if it's piped (not a TTY)."""
    if sys.stdin.isatty():
        return ''
    return sys.stdin.read().strip()


def _read_cache() -> Sequence[Mapping[str, object]] | None:
    """Read cached host discovery results if fresh enough."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get('timestamp', 0) > CACHE_TTL_SECONDS:
            return None
        return data.get('hosts', [])  # type: ignore[no-any-return]  # json.loads returns Any
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _write_cache(hosts: Sequence[DiscoveredHost]) -> None:
    """Write host discovery results to cache atomically.

    The explicit ``chmod 0o700`` on the directory matters: ``atomic_write``
    creates ``target.parent`` via ``mkdir(parents=True, exist_ok=True)`` which
    respects umask (typically 0o755). We tighten the directory after.
    The cache file itself is set to 0o600 by ``atomic_write``'s ``mode=`` kwarg.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CACHE_DIR, 0o700)
    entries = [{'alias': h.alias, 'hostname': h.hostname, 'ips': list(h.ips), 'port': h.port} for h in hosts]
    payload = json.dumps({'timestamp': time.time(), 'hosts': entries}).encode()
    atomic_write(CACHE_FILE, payload, mode=0o600)


# -- Error boundary handlers --------------------------------------------------
# Exception classes live in claude_remote_bash.exceptions. Handlers stay here
# because dispatch is a CLI-layer concern (how errors surface to the user).


@error_boundary.handler(RemoteBashError)
def _handle_user_facing(exc: RemoteBashError) -> None:
    """Every expected exception self-formats via ``__str__``; just print it."""
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    """Unexpected exceptions get the class name prepended for diagnostic clarity."""
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
