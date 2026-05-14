"""Daemon entry point: TCP server + mDNS registration + command dispatch.

The CLI is built on typer: bare invocation starts the daemon; setup actions
are subcommands. Typer auto-generates per-command ``--help`` and safely parses
argv.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
import typer.completion
from cc_lib.cli import add_completion_command, add_help_command
from cc_lib.error_boundary import ErrorBoundary

__all__ = [
    'main',
]

from claude_remote_bash.auth import DaemonConfig, generate_key, load_config, save_config, verify_key
from claude_remote_bash.context import SessionContextStore
from claude_remote_bash.discovery import register_service, unregister_service
from claude_remote_bash.exceptions import (
    ConfigError,
    FirewallApprovalError,
    LaunchdError,
    ProtocolError,
    RemoteBashError,
)
from claude_remote_bash.executor import execute_command
from claude_remote_bash.nfsd_manager import NfsdManager
from claude_remote_bash.protocol import (
    CONTROL_CHANNEL,
    parse_message,
    read_frame,
    read_message,
    write_frame,
    write_message,
)
from claude_remote_bash.schemas.protocol import (
    AuthFail,
    AuthOk,
    AuthRequest,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
    Message,
    MountRequest,
    MountResponse,
    TunnelOk,
    TunnelOpen,
    UnmountRequest,
    UnmountResponse,
)

logger = logging.getLogger(__name__)

VERSION = importlib.metadata.version('claude-remote-bash')
LAUNCHD_LABEL = 'com.claude-remote-bash.daemon'

error_boundary = ErrorBoundary(exit_code=1)

app = typer.Typer(
    help=(
        'Cross-machine shell daemon — TCP server + mDNS registration.\n\n'
        'Bare invocation starts the daemon. Use subcommands for setup:\n'
        '  init               Generate a new PSK and save config.\n'
        '  join KEY           Save a shared PSK from another machine.\n'
        "  set-name ALIAS     Set this daemon's alias.\n"
        '  allow-firewall     Approve daemon in macOS Application Firewall (sudo).\n'
        '  install-service    Install launchd LaunchAgent — run at login, auto-restart.\n'
        '  uninstall-service  Remove the launchd LaunchAgent.\n'
    ),
    add_completion=False,
    no_args_is_help=False,
    # Bind -h to --help. The daemon CLI doesn't go through cc_lib.create_app,
    # so it doesn't inherit the cascade — set explicitly here for consistency
    # with every other claude-workspace CLI.
    context_settings={'help_option_names': ['-h', '--help']},
)
typer.completion.completion_init()
add_completion_command(app)
add_help_command(app)


def main() -> None:
    """CLI entry point for claude-remote-bash-daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    app(prog_name='claude-remote-bash-daemon')


@app.command()
@error_boundary
def init(
    name: Annotated[str | None, typer.Option('--name', '-n', help="Set this daemon's alias during init.")] = None,
) -> None:
    """Generate a new PSK and save config."""
    key = generate_key()
    config = DaemonConfig(
        name=name or '',
        auth_key=key,
        shell=os.environ.get('SHELL', '/bin/zsh'),
    )
    path = save_config(config)

    try:
        subprocess.run(['pbcopy'], input=key.encode(), check=True)  # noqa: S603, S607 — pbcopy is a trusted macOS utility
        clipboard_msg = ' (copied to clipboard)'
    except (FileNotFoundError, subprocess.CalledProcessError):
        clipboard_msg = ''

    print(f'Initialized: {path}')
    print(f'Auth key: {key}{clipboard_msg}')
    if not name:
        print('Set a name: claude-remote-bash-daemon set-name <alias>')
    print(f'On other machines: claude-remote-bash-daemon join {key}')


@app.command()
@error_boundary
def join(
    key: Annotated[str, typer.Argument(help='Shared PSK printed by `init` on the first machine.')],
    name: Annotated[str | None, typer.Option('--name', '-n', help="Set this daemon's alias during join.")] = None,
) -> None:
    """Save a shared PSK from another machine."""
    config = load_config() or DaemonConfig()
    config.auth_key = key
    if name:
        config.name = name
    config.shell = os.environ.get('SHELL', '/bin/zsh')
    path = save_config(config)

    print(f'Joined: {path}')
    print('Auth key saved')
    if config.name:
        print(f'Name: {config.name}')
    else:
        print('Set a name: claude-remote-bash-daemon set-name <alias>')


@app.command(name='set-name')
@error_boundary
def set_name(
    alias: Annotated[str, typer.Argument(help="This daemon's alias (shown to peers via mDNS).")],
) -> None:
    """Set this daemon's alias."""
    config = load_config()
    if config is None:
        raise ConfigError('No config found. Run: claude-remote-bash-daemon init')
    config.name = alias
    save_config(config)
    print(f'Name set: {alias}')
    print('Start the daemon: claude-remote-bash-daemon')


@app.command(name='allow-firewall')
@error_boundary
def allow_firewall() -> None:
    """Approve daemon's python binary in the macOS Application Firewall (sudo).

    The daemon triggers an "Allow python3.13 to accept incoming connections"
    dialog on first run. Headless hosts or dismissed dialogs need this codified
    approval path. Targets ``sys.executable`` specifically — the binary the
    daemon actually runs under, not whatever ``python3`` resolves to on PATH.
    """
    if sys.platform != 'darwin':
        raise FirewallApprovalError(f'macOS-only (current platform: {sys.platform})')

    socketfilterfw = '/usr/libexec/ApplicationFirewall/socketfilterfw'
    if not Path(socketfilterfw).is_file():
        raise FirewallApprovalError(
            f'{socketfilterfw} not found — macOS Application Firewall is unavailable on this host.'
        )

    binary = sys.executable
    print(f'Approving binary in Application Firewall: {binary}')
    print('(will prompt for sudo)')

    for cmd in (['--add', binary], ['--unblockapp', binary]):
        result = subprocess.run(  # noqa: S603 — path is a fixed system utility
            ['sudo', socketfilterfw, *cmd],
            check=False,
        )
        if result.returncode != 0:
            raise FirewallApprovalError(f'socketfilterfw {cmd[0]} exited with code {result.returncode}')

    print('Approved. Restart of the daemon is NOT required — existing socket is unblocked.')


@app.command(name='install-service')
@error_boundary
def install_service() -> None:
    """Install a launchd LaunchAgent — daemon runs at login and auto-restarts.

    Writes ~/Library/LaunchAgents/<label>.plist and calls ``launchctl load``.
    Replaces any existing installation (unload + rewrite + load) so config
    changes propagate. Requires a complete config (auth_key + name).
    """
    if sys.platform != 'darwin':
        raise LaunchdError(f'macOS-only (current platform: {sys.platform})')

    config = load_config()
    if config is None or not config.auth_key or not config.name:
        raise ConfigError(
            'Cannot install service — config is incomplete.\n'
            'Run: claude-remote-bash-daemon init  (or join <key>)\n'
            '     claude-remote-bash-daemon set-name <alias>'
        )

    binary = _resolve_daemon_binary()
    plist_path = _launchd_plist_path()
    log_path = _launchd_log_path()

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    plist_path.write_text(_render_launchd_plist(binary=binary, log_path=log_path))

    # If already loaded, unload first so the rewritten plist takes effect.
    subprocess.run(  # noqa: S603, S607 — launchctl is a fixed system utility
        ['launchctl', 'unload', str(plist_path)],
        check=False,
        capture_output=True,
    )

    result = subprocess.run(  # noqa: S603, S607 — launchctl is a fixed system utility
        ['launchctl', 'load', str(plist_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LaunchdError(f'launchctl load exited with code {result.returncode}\n{result.stderr.strip()}')

    print(f'Installed: {plist_path}')
    print(f'Binary:    {binary}')
    print(f'Logs:      {log_path}')
    print(f'Status:    launchctl list | grep {LAUNCHD_LABEL}')


@app.command(name='uninstall-service')
@error_boundary
def uninstall_service() -> None:
    """Remove the launchd LaunchAgent and unload the daemon.

    Idempotent: reports cleanly if the plist is not installed.
    """
    if sys.platform != 'darwin':
        raise LaunchdError(f'macOS-only (current platform: {sys.platform})')

    plist_path = _launchd_plist_path()
    if not plist_path.exists():
        print(f'Not installed (no plist at {plist_path}).')
        return

    # Unload if loaded. Ignore non-zero exit — it just means it wasn't loaded.
    subprocess.run(  # noqa: S603, S607 — launchctl is a fixed system utility
        ['launchctl', 'unload', str(plist_path)],
        check=False,
        capture_output=True,
    )
    plist_path.unlink()
    print(f'Uninstalled: {plist_path}')


async def run_daemon(config: DaemonConfig) -> None:
    """Start the daemon: TCP server + mDNS registration."""
    daemon = _Daemon(config)
    handler_tasks: set[asyncio.Task[None]] = set()

    async def _tracked_handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError('asyncio.current_task() returned None inside server callback')
        handler_tasks.add(task)
        try:
            await daemon.handle_client(reader, writer)
        finally:
            handler_tasks.discard(task)

    server = await asyncio.start_server(_tracked_handle, '0.0.0.0', 0)
    port = server.sockets[0].getsockname()[1]
    logger.info('Listening on port %d', port)

    azc, info = await register_service(port, alias=config.name, version=VERSION)
    logger.info('Registered mDNS service: %s (alias=%s)', info.name, config.name)

    stop = asyncio.Event()

    def _signal_handler() -> None:
        logger.info('Shutting down...')
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    print(f'claude-remote-bash daemon v{VERSION}')
    print(f'  Alias: {config.name}')
    print(f'  Port:  {port}')
    print(f'  Shell: {config.shell}')
    print(f'  PID:   {os.getpid()}')

    async with server:
        await stop.wait()
        # Cancel client handlers before the context exits — Server.wait_closed()
        # (awaited by async-with's __aexit__) blocks until every active handler
        # finishes naturally, so a slow handler would otherwise pin shutdown.
        for t in handler_tasks:
            t.cancel()
        if handler_tasks:
            await asyncio.gather(*handler_tasks, return_exceptions=True)

    await daemon.shutdown_nfsd()
    await unregister_service(azc, info)
    logger.info('Shutdown complete')


class _Daemon:
    """TCP server that accepts authenticated connections and executes commands."""

    AUTH_TIMEOUT_SECONDS = 10.0

    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._contexts = SessionContextStore(default_cwd=os.path.expanduser('~'))
        self._nfsd_manager = NfsdManager()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single client TCP connection."""
        peer = writer.get_extra_info('peername')
        logger.info('Connection from %s', peer)

        try:
            if not await self._authenticate(reader, writer):
                return
            await self._dispatch_loop(reader, writer)
        except asyncio.IncompleteReadError:
            logger.info('Client %s disconnected', peer)
        except ProtocolError:
            logger.exception('Protocol error from %s', peer)
        finally:
            writer.close()
            await writer.wait_closed()

    async def shutdown_nfsd(self) -> None:
        """SIGTERM any crb-nfsd children spawned for filesystem mounts."""
        await self._nfsd_manager.shutdown()

    async def _authenticate(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        """Perform PSK authentication handshake. Returns True on success."""
        try:
            msg = await asyncio.wait_for(read_message(reader), timeout=self.AUTH_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning('Auth timeout from %s', writer.get_extra_info('peername'), exc_info=True)
            return False

        if not isinstance(msg, AuthRequest):
            await write_message(writer, AuthFail(reason='expected auth message'))
            return False

        if not verify_key(msg.key, self._config):
            logger.warning('Authentication failed from %s', writer.get_extra_info('peername'))
            await write_message(writer, AuthFail(reason='invalid key'))
            return False

        await write_message(
            writer,
            AuthOk(
                alias=self._config.name,
                hostname=socket.gethostname(),
                os='darwin',
                user=os.environ.get('USER', 'unknown'),
                shell=self._config.shell,
                version=VERSION,
            ),
        )
        logger.info('Authenticated: %s', writer.get_extra_info('peername'))
        return True

    async def _dispatch_loop(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Per-connection frame router.

        Channel 0 carries control-channel JSON messages; non-zero channels
        carry opaque tunnel payload routed to a per-channel ``crb-nfsd``
        loopback socket. Tunnel channels are allocated on demand by
        ``TunnelOpen`` and torn down when the connection closes or the peer
        sends a zero-length frame on the channel.

        A single ``write_lock`` serializes the multiple producers writing
        back on this connection (control responses + per-channel reverse
        pumps). A queue would give better fairness; the lock is correct
        and ~half the LOC for v0.1.
        """
        write_lock = asyncio.Lock()
        channels: dict[int, tuple[asyncio.StreamWriter, asyncio.Task[None]]] = {}
        next_channel_id = 1

        try:
            while True:
                channel_id, payload = await read_frame(reader)

                if channel_id == CONTROL_CHANNEL:
                    msg = parse_message(payload)
                    if isinstance(msg, TunnelOpen):
                        new_id = next_channel_id
                        next_channel_id += 1
                        entry = await self._open_tunnel(msg, new_id, writer, write_lock)
                        if entry is not None:
                            channels[new_id] = entry
                    else:
                        response = await self._handle_message(msg)
                        async with write_lock:
                            await write_message(writer, response)
                    continue

                entry = channels.get(channel_id)
                if entry is None:
                    logger.warning('frame for unknown channel_id=%d (dropped)', channel_id)
                    continue
                nfsd_writer, _pump = entry
                if not payload:
                    # Zero-length frame = client closed this channel.
                    nfsd_writer.close()
                    _nfsd_writer, pump_task = channels.pop(channel_id)
                    pump_task.cancel()
                else:
                    nfsd_writer.write(payload)
                    await nfsd_writer.drain()
        finally:
            for nfsd_writer, pump_task in channels.values():
                pump_task.cancel()
                nfsd_writer.close()
            pumps = [t for _, t in channels.values()]
            if pumps:
                await asyncio.wait(pumps)

    async def _handle_message(self, msg: Message) -> Message:
        """Dispatch a message to the appropriate handler."""
        if isinstance(msg, ExecuteRequest):
            return await self._handle_execute(msg)
        if isinstance(msg, MountRequest):
            return await self._handle_mount(msg)
        if isinstance(msg, UnmountRequest):
            return await self._handle_unmount(msg)
        return ErrorResponse(message=f'unexpected message type: {msg.type}')

    async def _handle_execute(self, msg: ExecuteRequest) -> ExecuteResult | ErrorResponse:
        """Execute a command with session context CWD tracking."""
        ctx = self._contexts.get(msg.session_id, msg.agent_id)

        result = await execute_command(
            msg.command,
            cwd=ctx.cwd,
            shell=self._config.shell,
            timeout=msg.timeout,
        )

        ctx.update_cwd(result.cwd)

        return ExecuteResult(
            id=msg.id,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            cwd=result.cwd,
        )

    async def _handle_mount(self, msg: MountRequest) -> MountResponse | ErrorResponse:
        """Spawn (or reuse) a crb-nfsd serving the requested root."""
        try:
            mount_id = await self._nfsd_manager.acquire(msg.root, msg.readonly)
        except RemoteBashError as exc:
            return ErrorResponse(id=msg.id, message=str(exc))
        return MountResponse(id=msg.id, mount_id=mount_id)

    async def _handle_unmount(self, msg: UnmountRequest) -> UnmountResponse:
        """Drop one hold on the mount; SIGTERM the child when refcount hits zero."""
        terminated = await self._nfsd_manager.release(msg.mount_id)
        return UnmountResponse(id=msg.id, mount_id=msg.mount_id, child_terminated=terminated)

    async def _open_tunnel(
        self,
        msg: TunnelOpen,
        channel_id: int,
        client_writer: asyncio.StreamWriter,
        write_lock: asyncio.Lock,
    ) -> tuple[asyncio.StreamWriter, asyncio.Task[None]] | None:
        """Allocate routing for a new tunnel channel against ``msg.mount_id``.

        Returns ``(nfsd_writer, pump_task)`` on success; the dispatch loop
        stores the pair in its per-channel registry. Inbound client frames
        on this channel are written to ``nfsd_writer``; the spawned pump
        task copies bytes the other way (nfsd reads → client frames).

        On failure, writes an ``ErrorResponse`` on the control channel and
        returns ``None``.
        """
        nfsd = self._nfsd_manager.get(msg.mount_id)
        if nfsd is None:
            async with write_lock:
                await write_message(client_writer, ErrorResponse(message=f'unknown mount_id: {msg.mount_id}'))
            return None

        try:
            nfsd_reader, nfsd_writer = await asyncio.open_connection('127.0.0.1', nfsd.port)
        except OSError as exc:
            async with write_lock:
                await write_message(client_writer, ErrorResponse(message=f'tunnel connect failed: {exc}'))
            return None

        async with write_lock:
            await write_message(client_writer, TunnelOk(mount_id=msg.mount_id, channel_id=channel_id))

        pump_task = asyncio.create_task(
            self._pump_nfsd_to_client(channel_id, nfsd_reader, client_writer, write_lock),
            name=f'tunnel-pump-ch{channel_id}',
        )
        return (nfsd_writer, pump_task)

    async def _pump_nfsd_to_client(
        self,
        channel_id: int,
        nfsd_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        write_lock: asyncio.Lock,
    ) -> None:
        """Copy bytes from the nfsd loopback socket back to the client as channel-N frames.

        Terminates on either nfsd-side EOF (sends a zero-length frame as
        close signal) or external cancellation (when the dispatch loop tears
        down the channel).
        """
        while True:
            chunk = await nfsd_reader.read(65536)
            async with write_lock:
                await write_frame(client_writer, channel_id, chunk)
            if not chunk:
                return


@app.callback(invoke_without_command=True)
@error_boundary
def _default(ctx: typer.Context) -> None:
    """Start the daemon on bare invocation; delegate to subcommand handlers otherwise."""
    if ctx.invoked_subcommand is not None:
        return
    config = _require_complete_config()
    asyncio.run(run_daemon(config))


def _require_complete_config() -> DaemonConfig:
    """Load the config or raise ConfigError with the next-step command."""
    config = load_config()
    if config is None:
        raise ConfigError('No config found. Run: claude-remote-bash-daemon init')
    if not config.auth_key:
        raise ConfigError(
            'No auth key configured. Run one of:\n'
            '  claude-remote-bash-daemon init          # new mesh (this machine mints the PSK)\n'
            '  claude-remote-bash-daemon join <key>    # existing mesh (paste PSK from another machine)'
        )
    if not config.name:
        raise ConfigError('No name configured. Run: claude-remote-bash-daemon set-name <alias>')
    return config


def _launchd_plist_path() -> Path:
    """Location of the LaunchAgent plist for the current user."""
    return Path.home() / 'Library' / 'LaunchAgents' / f'{LAUNCHD_LABEL}.plist'


def _launchd_log_path() -> Path:
    """Combined stdout+stderr log path for the launchd-managed daemon."""
    return Path.home() / 'Library' / 'Logs' / 'claude-remote-bash-daemon.log'


def _resolve_daemon_binary() -> Path:
    """Locate the ``claude-remote-bash-daemon`` binary on PATH.

    launchd runs with a minimal environment and no $PATH by default, so the
    plist must contain an absolute path. ``shutil.which`` resolves the shim
    installed by ``uv tool install`` (typically ``~/.local/bin/``).
    """
    which = shutil.which('claude-remote-bash-daemon')
    if which is None:
        raise LaunchdError(
            "Couldn't find `claude-remote-bash-daemon` on PATH.\n"
            'Install it first:\n'
            '  uv tool install --editable ~/claude-workspace/mcp/claude-remote-bash'
        )
    return Path(which).resolve()


def _render_launchd_plist(*, binary: Path, log_path: Path) -> str:
    """Render a LaunchAgent plist as a UTF-8 string.

    KeepAlive=true ensures launchd restarts the daemon if it crashes or the
    process is killed; RunAtLoad=true starts it on login (and at load time).
    Logs merge stdout+stderr because the daemon's logger writes to stderr and
    the banner/setup messages write to stdout — keeping them together in one
    file matches the terminal-running UX.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
"""


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
