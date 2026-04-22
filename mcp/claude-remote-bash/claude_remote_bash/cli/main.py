"""Command-line interface for claude-remote-bash."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

from claude_remote_bash.auth import load_config
from claude_remote_bash.discovery import DiscoveredHost, browse_hosts, resolve_host
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
    Message,
    ReadConfigRequest,
)
from claude_remote_bash.protocol import read_message, write_message

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

app = create_app(help='Cross-machine shell execution for Claude Code.')
add_completion_command(app)
error_boundary = ErrorBoundary(exit_code=1)

CACHE_DIR = Path.home() / '.claude-workspace' / 'claude-remote-bash'
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
    host: Annotated[str, typer.Option('--host', '-h', help='Host alias, hostname, or ip:port')] = '',
    session_id: Annotated[
        str, typer.Option('--session-id', envvar='CLAUDE_CODE_SESSION_ID', help='Session ID for CWD tracking')
    ] = 'default',
    agent_id: Annotated[
        str | None, typer.Option('--agent-id', envvar='CLAUDE_CODE_AGENT_ID', help='Agent ID for sub-agent isolation')
    ] = None,
    timeout: Annotated[float, typer.Option('--timeout', '-t', help='Command timeout in seconds')] = 120.0,
) -> None:
    """Execute a command on a remote host."""
    if not host:
        raise RemoteBashError('--host is required')

    cmd = command or _read_stdin()
    if not cmd:
        raise RemoteBashError('No command provided (pass as argument or pipe via stdin)')

    result = asyncio.run(
        _execute_remote(
            host=host,
            command=cmd,
            session_id=session_id,
            agent_id=agent_id,
            timeout=timeout,
        )
    )

    if result.stdout:
        typer.echo(result.stdout)
    if result.stderr:
        typer.echo(result.stderr, err=True)

    raise SystemExit(result.exit_code)


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


@app.command()
@error_boundary
def config(
    host: Annotated[str, typer.Option('--host', '-h', help='Host alias or hostname')] = '',
) -> None:
    """Read Claude Code configuration from a remote host."""
    if not host:
        raise RemoteBashError('--host is required')

    result = asyncio.run(_send_message(host, ReadConfigRequest()))
    typer.echo(json.dumps(result.model_dump(), indent=2))


# -- Async internals -----------------------------------------------------------


async def _execute_remote(
    *,
    host: str,
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
) -> ExecuteResult:
    """Connect to a daemon, authenticate, and execute a command."""
    ips, port = await _resolve_host(host)

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


async def _send_message(host: str, msg: Message) -> Message:
    """Connect, authenticate, send a message, and return the response."""
    ips, port = await _resolve_host(host)
    reader, writer = await _open_connection_any(ips, port)
    try:
        await _authenticate(reader, writer)
        await write_message(writer, msg)
        return await read_message(reader)
    finally:
        writer.close()
        await writer.wait_closed()


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
        raise AuthError('No auth key configured. Run: claude-remote-bash-daemon --init')

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
            '      claude-remote-bash-daemon --allow-firewall\n'
            '  - Verify on the target: does the daemon log show\n'
            '    "Connection from (ip, port)"? If yes the firewall is not the\n'
            "    cause; if no, it's confirmed."
        ) from exc

    if isinstance(response, AuthFail):
        raise AuthError(f'Authentication failed: {response.reason}')


async def _resolve_host(host: str) -> tuple[Sequence[str], int]:
    """Resolve a host specifier to (ips, port).

    A single host may advertise multiple IPv4 addresses (LAN + VPN + ethernet);
    the caller tries each in turn. For a literal ip:port the list has one entry.

    Resolution chain:
        1. Direct ip:port (e.g., "192.168.4.24:63276")
        2. Cached mDNS discovery (< 30s old)
        3. Fresh mDNS browse
    """
    if ':' in host:
        parts = host.rsplit(':', 1)
        return [parts[0]], int(parts[1])

    cached = _read_cache()
    if cached is not None:
        for entry in cached:
            if (
                str(entry.get('alias', '')).lower() == host.lower()
                or host.lower() in str(entry.get('hostname', '')).lower()
            ):
                return _cache_ips(entry), int(str(entry['port']))

    hosts = await browse_hosts(timeout=3.0)
    _write_cache(hosts)

    found = resolve_host(hosts, host)
    if found is None:
        raise HostNotFoundError(
            f'Host not found: {host}\n'
            'Run `claude-remote-bash discover` to see available hosts.\n'
            '\n'
            'Common causes:\n'
            "  - The daemon isn't running on the target.\n"
            '  - Alias typo — check the output of `discover`.\n'
            "  - Target is on a different LAN segment (mDNS doesn't cross subnets)."
        )

    return found.ips, found.port


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
    """Write host discovery results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(CACHE_DIR, 0o700)
    entries = [{'alias': h.alias, 'hostname': h.hostname, 'ips': list(h.ips), 'port': h.port} for h in hosts]
    CACHE_FILE.write_text(json.dumps({'timestamp': time.time(), 'hosts': entries}))
    os.chmod(CACHE_FILE, 0o600)


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
