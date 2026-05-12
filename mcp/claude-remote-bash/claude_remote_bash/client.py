"""Client-side wire interaction with a daemon: connect, authenticate, execute."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from claude_remote_bash.auth import load_config
from claude_remote_bash.exceptions import AuthError, DaemonError, HostUnreachableError
from claude_remote_bash.protocol import read_message, write_message
from claude_remote_bash.schemas.protocol import AuthFail, AuthRequest, ErrorResponse, ExecuteRequest, ExecuteResult

__all__ = [
    'authenticate',
    'execute_at',
    'open_connection_any',
]

CONNECT_TIMEOUT_SECONDS = 2.0
AUTH_TIMEOUT_SECONDS = 5.0


async def execute_at(
    *,
    ips: Sequence[str],
    port: int,
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
) -> ExecuteResult:
    """Run a command on the daemon at ``(ips, port)`` — connect, authenticate, execute."""
    reader, writer = await open_connection_any(ips, port)
    try:
        await authenticate(reader, writer)

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


async def open_connection_any(
    ips: Sequence[str],
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Try each IP with a short per-attempt timeout; return the first connection.

    Daemons advertise multiple IPv4 addresses (LAN + VPN + ethernet). The
    short per-attempt timeout means an unreachable address fails fast rather
    than hanging the whole invocation.
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


async def authenticate(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Perform PSK authentication on an open connection.

    The read is timed so that a TCP-connect-but-no-auth-response state — the
    macOS Application Firewall signature, where the kernel accepts the
    handshake but the firewall silently blocks the daemon from receiving
    the payload — surfaces as an actionable error instead of an indefinite
    hang.
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


def _reachability_hint(errors: Sequence[tuple[str, Exception]]) -> str:
    """Return a user-actionable hint based on the exception signatures of the connect attempts."""
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
