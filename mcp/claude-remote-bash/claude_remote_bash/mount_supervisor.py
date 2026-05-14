"""Detached supervisor process owning a single ``crb mount``.

Lifecycle: started by ``crb mount`` via ``subprocess.Popen`` with
``start_new_session=True``. Performs the mount, prints exactly one
``READY <mount_id>`` line to stdout (the spawning parent waits for it),
then blocks until SIGTERM, which triggers ``crb_mount_blocking``'s
``UnmountRequest`` + ``umount`` teardown.

Stdout is reserved for the success protocol — one ``READY`` line, nothing
else. All errors flow to stderr via the error boundary; the parent reads
stderr only when the supervisor exits before READY. All post-READY
logging goes to ``~/.crb/log/supervisor-<pid>.log``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary

from claude_remote_bash.exceptions import RemoteBashError
from claude_remote_bash.mount import crb_mount_blocking
from claude_remote_bash.mounts_registry import MountEntry, add_entry, remove_entry

__all__ = [
    'main',
]

LOG_DIR = Path.home() / '.crb' / 'log'
"""Per-supervisor log files live here; one file per PID."""

error_boundary = ErrorBoundary(exit_code=1)


def main() -> None:
    """Entry point — parse argv, configure logging, run the mount, signal READY."""
    _run()


@error_boundary
def _run() -> None:
    args = _parse_args()
    _configure_logging(args.mountpoint)
    asyncio.run(_supervise(args))


async def _supervise(args: argparse.Namespace) -> None:
    mountpoint = Path(args.mountpoint)
    pid = os.getpid()

    def on_mounted(mount_id: str) -> None:
        add_entry(
            MountEntry(
                mount_id=mount_id,
                peer_alias=args.peer_alias,
                remote_path=args.remote_path,
                mountpoint=str(mountpoint),
                supervisor_pid=pid,
                readonly=args.readonly,
                established_at=datetime.now(UTC),
            )
        )
        sys.stdout.write(f'READY {mount_id}\n')
        sys.stdout.flush()

    try:
        await crb_mount_blocking(
            peer_alias=args.peer_alias,
            remote_path=args.remote_path,
            mountpoint=mountpoint,
            readonly=args.readonly,
            on_mounted=on_mounted,
        )
    finally:
        remove_entry(mountpoint)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='crb-mount-supervisor',
        description='Internal: detached supervisor for a `crb mount`. Not for direct invocation.',
    )
    parser.add_argument('peer_alias')
    parser.add_argument('remote_path')
    parser.add_argument('mountpoint')
    parser.add_argument('--readonly', action='store_true')
    return parser.parse_args()


def _configure_logging(mountpoint: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'supervisor-{os.getpid()}.log'
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    logging.getLogger(__name__).info('Supervisor pid=%d mountpoint=%s', os.getpid(), mountpoint)


@error_boundary.handler(RemoteBashError)
def _handle_user_facing(exc: RemoteBashError) -> None:
    """User-facing errors self-format via ``__str__``; the parent reads this off stderr."""
    print(exc, file=sys.stderr)


@error_boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    """Unexpected exceptions: log the traceback to file, surface class+message on stderr."""
    logging.getLogger(__name__).exception('Supervisor crashed')
    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
