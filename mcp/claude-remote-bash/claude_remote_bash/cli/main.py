"""Command-line interface for claude-remote-bash."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Sequence
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import OutputFormat

from claude_remote_bash.cache import HostsCache
from claude_remote_bash.discovery import browse_hosts
from claude_remote_bash.dispatch import DispatchResult, DispatchService, HostRunResult
from claude_remote_bash.exceptions import RemoteBashError
from claude_remote_bash.schemas.discovery import DiscoveredHostInfo, DiscoverResult

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

app = create_app(help='Cross-machine shell execution for Claude Code.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


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
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Execute a command on one or more remote hosts.

    \b
    The target selector accepts:
      - a host alias (e.g. M2)
      - a comma-separated list (M2,M3,M4)
      - a named group from client_config.json (e.g. fleet)
      - a literal ip:port (192.168.4.22:51648)
      - any mix of the above; whitespace per-atom is stripped, duplicates rejected.

    With ``--format json``, emits a single ``DispatchResult`` JSON object on
    stdout — per-host stdout/stderr/exit_code/duration are captured in the
    result rather than streamed, so downstream parsers aren't corrupted by
    multi-host fan-out. The overall exit code still propagates.
    """
    if not target:
        raise RemoteBashError('--target is required')

    cmd = command or _read_stdin()
    if not cmd:
        raise RemoteBashError('No command provided (pass as argument or pipe via stdin)')

    asyncio.run(_drive_dispatch(target, cmd, session_id, agent_id, timeout, format))


@app.command()
@error_boundary
def discover(
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Browse the LAN for claude-remote-bash daemons.

    With ``--format json``, emits ``{"daemons": [...]}`` — wrapped in an
    object so future fields (errors, scan duration) can be added without
    breaking consumers. Empty result is ``{"daemons": []}``.
    """
    hosts = asyncio.run(browse_hosts(timeout=3.0))
    HostsCache.from_browse(hosts).write()

    if format == 'json':
        result = DiscoverResult(
            daemons=[
                DiscoveredHostInfo(
                    alias=h.alias,
                    hostname=h.hostname,
                    ips=list(h.ips),
                    port=h.port,
                    version=h.version,
                    is_self=h.is_self,
                )
                for h in hosts
            ]
        )
        typer.echo(result.model_dump_json())
        return

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
        self_marker = '  (self)' if h.is_self else ''
        typer.echo(f'  {h.alias:<12} {ips_str}:{h.port}  ({h.hostname})  v{h.version}{self_marker}')


# -- Output formatting ---------------------------------------------------------


async def _drive_dispatch(
    target: str,
    command: str,
    session_id: str,
    agent_id: str | None,
    timeout: float,
    output_format: OutputFormat,
) -> None:
    """Invoke ``DispatchService.run_target`` and format the result for the terminal.

    Text mode: single-host streams raw stdout/stderr; multi-host prefixes every
    line with ``[host]`` and prints a summary table.

    JSON mode: emits a single ``DispatchResult`` JSON object on stdout regardless
    of host count; per-host output is captured inside the result. Overall exit
    code propagates either way.
    """
    result = await DispatchService().run_target(
        target,
        command,
        session_id=session_id,
        agent_id=agent_id,
        timeout=timeout,
    )
    if output_format == 'json':
        typer.echo(result.model_dump_json())
        raise SystemExit(result.overall_exit_code)
    if len(result.results) == 1:
        _emit_single(result.results[0])
    else:
        _emit_multi(result)


def _emit_single(host: HostRunResult) -> None:
    """Stream one host's output exactly as if ``execute_at`` had been called directly."""
    if host.error is not None:
        raise RemoteBashError(host.error)
    if host.stdout:
        typer.echo(host.stdout)
    if host.stderr:
        typer.echo(host.stderr, err=True)
    raise SystemExit(host.exit_code)


def _emit_multi(result: DispatchResult) -> None:
    """Prefix each host's lines with ``[host]``, then print the summary table."""
    for r in result.results:
        for line in r.stdout.splitlines():
            typer.echo(f'[{r.host}] {line}')
        for line in r.stderr.splitlines():
            typer.echo(f'[{r.host}] {line}', err=True)
    _print_summary(result.results)
    raise SystemExit(result.overall_exit_code)


def _print_summary(results: Sequence[HostRunResult]) -> None:
    """Print a per-host summary table after multi-host dispatch."""
    width = max(len('host'), *(len(r.host) for r in results))
    typer.echo('')
    typer.echo(f'{"host":<{width}}  status  exit  duration')
    for r in results:
        status = 'ok' if r.error is None and r.exit_code == 0 else 'failed'
        suffix = f'  ({r.error})' if r.error else ''
        typer.echo(f'{r.host:<{width}}  {status:<6}  {r.exit_code:<4}  {r.duration_s:.2f}s{suffix}')


# -- Helpers -------------------------------------------------------------------


def _read_stdin() -> str:
    """Read command from stdin if it's piped (not a TTY)."""
    if sys.stdin.isatty():
        return ''
    return sys.stdin.read().strip()


# -- Error boundary handlers --------------------------------------------------


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
