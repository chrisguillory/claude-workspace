from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence, Set
from pathlib import Path
from typing import Annotated

import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import OutputFormat

from claude_remote_bash.cache import HostsCache
from claude_remote_bash.cli.output import DiscoverResult, GroupInfo
from claude_remote_bash.client_config import ClientConfig
from claude_remote_bash.discovery import DiscoveredHost, browse_hosts
from claude_remote_bash.dispatch import DispatchResult, DispatchService, HostRunResult
from claude_remote_bash.exceptions import ConfigError, RemoteBashError
from claude_remote_bash.mount import (
    crb_mount_blocking,
    default_mountpoint,
    spawn_detached_supervisor,
    supervisor_alive,
    terminate_supervisor,
)
from claude_remote_bash.mounts_registry import read_registry
from claude_remote_bash.paths import CLIENT_CONFIG

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
def mount(
    target: Annotated[
        str,
        typer.Argument(help='Mount target as `<peer>:<remote_path>` (e.g. `m2:~/projects/foo`)'),
    ],
    mountpoint: Annotated[
        Path | None,
        typer.Argument(help='Local mountpoint. Defaults to ~/.crb/host/<peer>/<basename>/.'),
    ] = None,
    readonly: Annotated[bool, typer.Option('--readonly', help='Mount read-only.')] = False,
    browse: Annotated[
        bool,
        typer.Option(
            '--browse',
            help='Show the mount in Finder\'s "Computer" view and sidebar. Off by default — '
            'enables Finder drag-and-drop at the cost of an eject button that bypasses `crb umount`.',
        ),
    ] = False,
    foreground: Annotated[
        bool,
        typer.Option(
            '--foreground',
            help='Block in the foreground instead of detaching. ^C unmounts. Useful for debugging.',
        ),
    ] = False,
) -> None:
    """Mount a remote directory over NFSv3.

    Detaches a supervisor process by default — ``crb umount <mountpoint>``
    cleans up. With ``--foreground``, blocks until ^C and then unmounts.
    The mount is hidden from Finder's volume listing by default; pass
    ``--browse`` if you want drag-and-drop visibility.

    \b
    Example:
      crb mount m2:~/projects/foo
      crb mounts                              # see live mounts
      crb umount ~/.crb/host/m2/foo           # tear it down
    """
    peer, sep, remote_path = target.partition(':')
    if not sep or not peer or not remote_path:
        raise RemoteBashError('mount target must be `<peer>:<remote_path>`')

    # ``resolve()`` absolutizes and follows parent symlinks (e.g. /tmp →
    # /private/tmp on macOS) so the registry's mountpoint matches what
    # ``terminate_supervisor`` later resolves the user's argument to.
    resolved_mountpoint = (mountpoint or default_mountpoint(peer, remote_path)).resolve()

    if foreground:
        asyncio.run(
            crb_mount_blocking(
                peer_alias=peer,
                remote_path=remote_path,
                mountpoint=resolved_mountpoint,
                readonly=readonly,
                browse=browse,
            )
        )
        return

    pid, mount_id = spawn_detached_supervisor(
        peer_alias=peer,
        remote_path=remote_path,
        mountpoint=resolved_mountpoint,
        readonly=readonly,
        browse=browse,
    )
    typer.echo(f'mounted {peer}:{remote_path} at {resolved_mountpoint}')
    typer.echo(f'supervisor pid={pid} mount_id={mount_id[:8]}')
    typer.echo(f'tear down with: crb umount {resolved_mountpoint}')


@app.command()
@error_boundary
def umount(
    mountpoint: Annotated[Path, typer.Argument(help='Mountpoint to tear down')],
) -> None:
    """Tear down a mount established via ``crb mount``.

    SIGTERMs the supervisor for the mount; the supervisor unmounts and
    removes its registry entry. Reports an error if no such mount exists
    or if the supervisor doesn't unwind cleanly.
    """
    terminate_supervisor(mountpoint)
    typer.echo(f'unmounted {mountpoint}')


@app.command()
@error_boundary
def mounts(
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """List active mounts established via ``crb mount``.

    Each entry's ``status`` is ``live`` if the supervisor PID still
    responds to ``kill -0``, otherwise ``orphan`` (registry left a stale
    entry — running ``crb umount`` against the mountpoint reaps it).
    """
    registry = read_registry()

    if format == 'json':
        typer.echo(registry.model_dump_json(by_alias=True))
        return

    if not registry.mounts:
        typer.echo('No active mounts.')
        return

    width = max(len('mountpoint'), *(len(m.mountpoint) for m in registry.mounts))
    typer.echo(f'{"mountpoint":<{width}}  status  pid       peer:remote')
    for m in registry.mounts:
        status = 'live' if supervisor_alive(m.supervisor_pid) else 'orphan'
        typer.echo(f'{m.mountpoint:<{width}}  {status:<6}  {m.supervisor_pid:<8}  {m.peer_alias}:{m.remote_path}')


@app.command()
@error_boundary
def discover(
    format: Annotated[OutputFormat, typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Browse the LAN for claude-remote-bash daemons and list configured groups.

    The text output renders three blocks: the local daemon (if its mDNS
    advertisement is on the wire), remote daemons, and named host groups
    from ``client_config.json``. With ``--format json``, emits a
    ``DiscoverResult`` envelope: ``{remote_daemons, local_daemon, groups}``.

    A malformed ``client_config.json`` does not fail the command; the error
    is logged to stderr and the daemons block still prints.
    """
    result = asyncio.run(browse_hosts(timeout=3.0))
    HostsCache.from_browse(result).write()
    groups = _load_groups_for_discover()

    if format == 'json':
        envelope = DiscoverResult(
            remote_daemons=result.remote_daemons,
            local_daemon=result.local_daemon,
            groups=[GroupInfo(name=name, members=list(members)) for name, members in groups.items()],
        )
        typer.echo(envelope.model_dump_json())
        return

    _render_daemons_block(result.remote_daemons, result.local_daemon)
    _render_groups_block(groups, result.remote_daemons, result.local_daemon)


def _load_groups_for_discover() -> Mapping[str, Sequence[str]]:
    """Load ``client_config.json`` groups for the discover command.

    A malformed config emits a warning to stderr and returns no groups —
    discover's primary job is to show daemons, so a bad config shouldn't
    fail the command.
    """
    try:
        return dict(ClientConfig.load().groups)
    except ConfigError as exc:
        typer.echo(f'Warning: {exc}', err=True)
        return {}


def _render_daemons_block(
    remote_daemons: Sequence[DiscoveredHost],
    local_daemon: DiscoveredHost | None,
) -> None:
    """Render the local daemon (if any) followed by the remote daemons."""
    if not remote_daemons and local_daemon is None:
        typer.echo('No daemons found on the network.')
        typer.echo('')
        typer.echo('If a daemon should be visible:')
        typer.echo('  - Verify the daemon is running on the target:')
        typer.echo('      `pgrep -f claude-remote-bash-daemon`')
        typer.echo('  - Verify network reachability: `ping <target>.local`')
        typer.echo("  - Ensure client and target are on the same LAN segment (mDNS doesn't cross subnets).")
        return

    total = len(remote_daemons) + (1 if local_daemon else 0)
    typer.echo(f'Found {total} daemon(s):\n')
    if local_daemon is not None:
        typer.echo(_format_host_line(local_daemon, is_local=True))
    for h in remote_daemons:
        typer.echo(_format_host_line(h, is_local=False))


def _format_host_line(host: DiscoveredHost, *, is_local: bool) -> str:
    """Render one host row: ``alias  ip(kind),ip(kind):port  (hostname)  vX.Y.Z  (self)?  (legacy)?``."""
    addrs = ','.join(f'{a.ip}({a.kind})' for a in host.addresses) if host.addresses else '?'
    markers = ''
    if is_local:
        markers += '  (self)'
    if host.legacy:
        markers += '  (legacy)'
    return f'  {host.alias:<12} {addrs}:{host.port}  ({host.hostname})  v{host.version}{markers}'


def _render_groups_block(
    groups: Mapping[str, Sequence[str]],
    remote_daemons: Sequence[DiscoveredHost],
    local_daemon: DiscoveredHost | None,
) -> None:
    """Render the configured-groups block after the daemons block.

    Member aliases not in the discovered host set are decorated with
    ``(no daemon)`` to surface stale config without failing the command.
    """
    typer.echo('')
    if not groups:
        typer.echo('No groups configured.')
        typer.echo(f'Define groups in {CLIENT_CONFIG} to target multiple hosts at once.')
        return

    discovered_aliases = {h.alias.lower() for h in remote_daemons}
    if local_daemon is not None:
        discovered_aliases.add(local_daemon.alias.lower())
    name_width = max(len(name) for name in groups)

    typer.echo(f'Groups ({len(groups)}):\n')
    for name, members in groups.items():
        # Bare commas, no spaces — spaces would shell-split the list, so the
        # no-space form pastes straight after `--target` as a single argument.
        rendered = ','.join(_decorate_member(m, discovered_aliases) for m in members)
        typer.echo(f'  {name:<{name_width}}  {rendered}')
    typer.echo('')
    typer.echo('Use with `execute --target <group>`.')


def _decorate_member(alias: str, discovered: Set[str]) -> str:
    """Return ``alias`` with a ``(no daemon)`` marker if not in ``discovered``."""
    if alias.lower() not in discovered:
        return f'{alias} (no daemon)'
    return alias


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
