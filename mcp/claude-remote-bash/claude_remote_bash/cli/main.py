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
from claude_remote_bash.models import (
    AuthFail,
    AuthRequest,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
    Message,
    ReadConfigRequest,
)
from claude_remote_bash.protocol import ProtocolError, read_message, write_message

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

    raise SystemExit(result.exit_code)


@app.command()
@error_boundary
def discover() -> None:
    """Browse the LAN for claude-remote-bash daemons."""
    hosts = asyncio.run(browse_hosts(timeout=3.0))
    _write_cache(hosts)

    if not hosts:
        typer.echo('No daemons found on the network.')
        return

    typer.echo(f'Found {len(hosts)} daemon(s):\n')
    for h in hosts:
        typer.echo(f'  {h.alias:<12} {h.ip}:{h.port}  ({h.hostname})  v{h.version}')


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
    ip, port = await _resolve_host(host)

    reader, writer = await asyncio.open_connection(ip, port)
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
    ip, port = await _resolve_host(host)
    reader, writer = await asyncio.open_connection(ip, port)
    try:
        await _authenticate(reader, writer)
        await write_message(writer, msg)
        return await read_message(reader)
    finally:
        writer.close()
        await writer.wait_closed()


async def _authenticate(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Perform PSK authentication."""
    cfg = load_config()
    if cfg is None or not cfg.auth_key:
        raise AuthError('No auth key configured. Run: claude-remote-bash-daemon --init')

    await write_message(writer, AuthRequest(key=cfg.auth_key))
    response = await read_message(reader)

    if isinstance(response, AuthFail):
        raise AuthError(f'Authentication failed: {response.reason}')


async def _resolve_host(host: str) -> tuple[str, int]:
    """Resolve a host specifier to (ip, port).

    Resolution chain:
        1. Direct ip:port (e.g., "192.168.4.24:63276")
        2. Cached mDNS discovery (< 30s old)
        3. Fresh mDNS browse
    """
    if ':' in host:
        parts = host.rsplit(':', 1)
        return parts[0], int(parts[1])

    cached = _read_cache()
    if cached is not None:
        for entry in cached:
            if str(entry.get('alias', '')).lower() == host.lower():
                return str(entry['ip']), int(str(entry['port']))
            if host.lower() in str(entry.get('hostname', '')).lower():
                return str(entry['ip']), int(str(entry['port']))

    hosts = await browse_hosts(timeout=3.0)
    _write_cache(hosts)

    found = resolve_host(hosts, host)
    if found is None:
        raise HostNotFoundError(f'Host not found: {host}\nRun `claude-remote-bash discover` to see available hosts.')

    return found.ip, found.port


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
    entries = [{'alias': h.alias, 'hostname': h.hostname, 'ip': h.ip, 'port': h.port} for h in hosts]
    CACHE_FILE.write_text(json.dumps({'timestamp': time.time(), 'hosts': entries}))
    os.chmod(CACHE_FILE, 0o600)


# -- Exceptions + error boundary handlers -------------------------------------


class RemoteBashError(Exception):
    """Base exception for CLI errors."""


class AuthError(RemoteBashError):
    """Authentication failed or not configured."""


class HostNotFoundError(RemoteBashError):
    """Host alias could not be resolved via mDNS."""


class DaemonError(RemoteBashError):
    """Daemon returned an error response."""


@error_boundary.handler(AuthError)
def _handle_auth_error(exc: AuthError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(HostNotFoundError)
def _handle_host_not_found(exc: HostNotFoundError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(DaemonError)
def _handle_daemon_error(exc: DaemonError) -> None:
    print(f'Daemon error: {exc}', file=sys.stderr)


@error_boundary.handler(ConnectionRefusedError)
def _handle_connection_refused(exc: ConnectionRefusedError) -> None:
    print(f'Connection refused — is the daemon running? ({exc})', file=sys.stderr)


@error_boundary.handler(ProtocolError)
def _handle_protocol_error(exc: ProtocolError) -> None:
    print(f'Protocol error: {exc}', file=sys.stderr)


@error_boundary.handler(RemoteBashError)
def _handle_remote_bash_error(exc: RemoteBashError) -> None:
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
