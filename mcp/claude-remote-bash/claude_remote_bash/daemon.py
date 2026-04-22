"""Daemon entry point: TCP server + mDNS registration + command dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    'main',
]

from cc_lib.error_boundary import ErrorBoundary

from claude_remote_bash.auth import DaemonConfig, generate_key, load_config, save_config, verify_key
from claude_remote_bash.context import SessionContextStore
from claude_remote_bash.discovery import register_service, unregister_service
from claude_remote_bash.exceptions import (
    ConfigError,
    FirewallApprovalError,
    ProtocolError,
    RemoteBashError,
)
from claude_remote_bash.executor import execute_command
from claude_remote_bash.models import (
    AuthFail,
    AuthOk,
    AuthRequest,
    ConfigContent,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
    Message,
    ReadConfigRequest,
)
from claude_remote_bash.protocol import read_message, write_message

logger = logging.getLogger(__name__)

VERSION = '0.1.0'

error_boundary = ErrorBoundary(exit_code=1)


@error_boundary
def main() -> None:
    """CLI entry point for claude-remote-bash-daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )

    args = sys.argv[1:]

    if '--init' in args:
        _cmd_init()
        return

    if '--join' in args:
        idx = args.index('--join')
        if idx + 1 >= len(args):
            raise ConfigError('Usage: claude-remote-bash-daemon --join <key>')
        _cmd_join(args[idx + 1])
        return

    if '--allow-firewall' in args:
        _cmd_allow_firewall()
        return

    name = _extract_flag(args, '--name')

    config = load_config()
    if config is None:
        raise ConfigError('No config found. Run: claude-remote-bash-daemon --init')

    # `--name <alias>` is a config-only operation that exits after saving,
    # matching the UX of --init and --join. Starting the daemon is always
    # the bare invocation (no flags).
    if name:
        config.name = name
        save_config(config)
        print(f'Name set: {name}')
        print('Start the daemon: claude-remote-bash-daemon')
        return

    if not config.auth_key:
        raise ConfigError('No auth key configured. Run: claude-remote-bash-daemon --init')

    if not config.name:
        raise ConfigError('No name configured. Run: claude-remote-bash-daemon --name <alias>')

    asyncio.run(run_daemon(config))


async def run_daemon(config: DaemonConfig) -> None:
    """Start the daemon: TCP server + mDNS registration."""
    daemon = _Daemon(config)

    server = await asyncio.start_server(daemon.handle_client, '0.0.0.0', 0)
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

    await unregister_service(azc, info)
    logger.info('Shutdown complete')


class _Daemon:
    """TCP server that accepts authenticated connections and executes commands."""

    AUTH_TIMEOUT_SECONDS = 10.0

    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._contexts = SessionContextStore(default_cwd=os.path.expanduser('~'))

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
        """Read and dispatch messages until the client disconnects."""
        while True:
            msg = await read_message(reader)
            response = await self._handle_message(msg)
            await write_message(writer, response)

    async def _handle_message(self, msg: Message) -> Message:
        """Dispatch a message to the appropriate handler."""
        if isinstance(msg, ExecuteRequest):
            return await self._handle_execute(msg)
        if isinstance(msg, ReadConfigRequest):
            return self._handle_read_config()
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
            exit_code=result.exit_code,
            cwd=result.cwd,
        )

    def _handle_read_config(self) -> ConfigContent:
        """Read Claude Code configuration files."""
        return ConfigContent(
            claude_json=json.loads((Path.home() / '.claude.json').read_text()),
            settings_json=json.loads((Path.home() / '.claude' / 'settings.json').read_text()),
        )


def _cmd_init() -> None:
    """Generate a new PSK and save config."""
    key = generate_key()
    name_flag = _extract_flag(sys.argv, '--name')

    config = DaemonConfig(
        name=name_flag or '',
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
    if not name_flag:
        print('Set a name: claude-remote-bash-daemon --name <alias>')
    print(f'On other machines: claude-remote-bash-daemon --join {key}')


def _cmd_join(key: str) -> None:
    """Join an existing cluster by saving the shared key."""
    name_flag = _extract_flag(sys.argv, '--name')

    config = load_config() or DaemonConfig()
    config.auth_key = key
    if name_flag:
        config.name = name_flag
    config.shell = os.environ.get('SHELL', '/bin/zsh')
    path = save_config(config)

    print(f'Joined: {path}')
    print('Auth key saved (from --join)')
    if config.name:
        print(f'Name: {config.name}')
    else:
        print('Set a name: claude-remote-bash-daemon --name <alias>')


def _cmd_allow_firewall() -> None:
    """Approve this daemon's python binary in the macOS Application Firewall.

    On first run the daemon triggers an "Allow python3.13 to accept incoming
    connections" dialog. Users who dismiss it or run headless need a
    codified way to grant approval — this flag wraps the two required
    ``socketfilterfw`` calls against ``sys.executable`` (the actual binary
    the daemon runs under, not whatever ``python3`` resolves to in $PATH).

    Requires sudo; macOS-only.
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


def _extract_flag(args: Sequence[str], flag: str) -> str | None:
    """Extract a flag value from argv-style args. Returns None if not present."""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    return None


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
