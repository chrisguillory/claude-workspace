from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from claude_remote_bash.exceptions import DaemonError

__all__ = [
    'NfsdManager',
    'NfsdProcess',
]

logger = logging.getLogger(__name__)

NFSD_BINARY = 'crb-nfsd'
"""Maturin (``bindings = "bin"``) installs this into the same venv as the
daemon, so ``Path(sys.executable).parent / NFSD_BINARY`` is the canonical
location. ``PATH`` lookup is the fallback for unusual invocations."""

READY_TIMEOUT_SECONDS = 10.0
"""How long to wait for the spawned binary to print `LISTEN_PORT=N` + `READY`."""

TERMINATE_TIMEOUT_SECONDS = 3.0
"""Grace window between SIGTERM and SIGKILL during shutdown."""


@dataclass
class NfsdProcess:
    """A running ``crb-nfsd`` child serving a single ``(root, readonly)``."""

    root: Path
    """Canonical absolute path the child is serving."""

    readonly: bool

    port: int
    """The loopback port the child is listening on (printed via LISTEN_PORT=N)."""

    process: asyncio.subprocess.Process

    refcount: int
    """Number of live ``mount_id`` claims on this child. Zero => terminate."""


class NfsdManager:
    """Spawns ``crb-nfsd`` children and shares them across same-key holders.

    Two mount_ids that resolve to the same ``(canonical(root), readonly)``
    share a single child. Each holder gets its own ``mount_id`` (UUID) and
    can independently release; the child is SIGTERMed only when the last
    holder releases.

    Concrete reason: two Claude tabs on the same machine mounting the same
    peer path. Without refcount, tab A's ``release`` would kill the child
    while tab B is mid-RPC.

    No exit-watcher: if a child dies unexpectedly (OOM, bug), the registry
    entry stays but its loopback port is gone. The next tunnel claim hits
    ``ECONNREFUSED`` and surfaces the error to the client. Proactive
    eviction would shave a few ms off the failure window for ~30 LOC of
    background-task plumbing — not worth it.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, NfsdProcess] = {}
        self._by_key: dict[tuple[Path, bool], NfsdProcess] = {}
        # Serializes the check-spawn-insert critical section in acquire so
        # two concurrent calls with the same key don't both miss the dedup
        # and both spawn (the second would orphan the first child).
        self._lock = asyncio.Lock()

    async def acquire(self, root: str, readonly: bool) -> str:
        """Get or spawn an nfsd for ``(canonical(root), readonly)``. Returns the new mount_id.

        ``--block-prefix $HOME/.crb/`` is always passed so the served tree
        can't loop back through the supervisor's own state directory.
        """
        canonical = Path(root).expanduser().resolve()
        if not canonical.is_dir():
            raise DaemonError(f'root is not an existing directory: {canonical}')

        key = (canonical, readonly)
        mount_id = uuid.uuid4().hex

        async with self._lock:
            existing = self._by_key.get(key)
            if existing is not None:
                existing.refcount += 1
                self._by_id[mount_id] = existing
                logger.info(
                    'Sharing nfsd: root=%s readonly=%s mount_id=%s (holders=%d)',
                    canonical,
                    readonly,
                    mount_id,
                    existing.refcount,
                )
                return mount_id

            nfsd = await self._spawn(canonical, readonly)
            self._by_id[mount_id] = nfsd
            self._by_key[key] = nfsd
            logger.info(
                'Spawned nfsd: root=%s readonly=%s mount_id=%s port=%d',
                canonical,
                readonly,
                mount_id,
                nfsd.port,
            )
            return mount_id

    async def release(self, mount_id: str) -> bool:
        """Drop one hold on ``mount_id``. Returns True iff this released the last holder.

        Idempotent — releasing an unknown mount_id returns False.
        """
        nfsd = self._by_id.pop(mount_id, None)
        if nfsd is None:
            return False
        nfsd.refcount -= 1
        if nfsd.refcount > 0:
            logger.info(
                'Released hold: mount_id=%s remaining=%d',
                mount_id,
                nfsd.refcount,
            )
            return False
        self._by_key.pop((nfsd.root, nfsd.readonly), None)
        await self._terminate(nfsd)
        logger.info('Terminated nfsd: last holder mount_id=%s released', mount_id)
        return True

    def get(self, mount_id: str) -> NfsdProcess | None:
        """Look up the running nfsd by mount_id, or None if unknown."""
        return self._by_id.get(mount_id)

    async def shutdown(self) -> None:
        """SIGTERM all children and await their exit. Used by daemon shutdown."""
        procs = list(self._by_key.values())
        self._by_id.clear()
        self._by_key.clear()
        for nfsd in procs:
            await self._terminate(nfsd)
        if procs:
            logger.info('Shut down %d nfsd child(ren)', len(procs))

    async def _spawn(self, root: Path, readonly: bool) -> NfsdProcess:
        binary = _locate_nfsd_binary()

        crb_dir = Path.home() / '.crb'
        crb_dir.mkdir(parents=True, exist_ok=True)

        args = [
            binary,
            '--root',
            str(root),
            '--listen',
            '127.0.0.1:0',
            '--block-prefix',
            str(crb_dir),
        ]
        if readonly:
            args.append('--readonly')

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            port = await asyncio.wait_for(_read_listen_port(process), timeout=READY_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            process.terminate()
            await process.wait()
            raise DaemonError(f'{NFSD_BINARY} did not signal READY within {READY_TIMEOUT_SECONDS:.0f}s') from exc

        return NfsdProcess(
            root=root,
            readonly=readonly,
            port=port,
            process=process,
            refcount=1,
        )

    async def _terminate(self, nfsd: NfsdProcess) -> None:
        """SIGTERM the child; SIGKILL if it doesn't exit in time."""
        if nfsd.process.returncode is not None:
            return
        try:
            nfsd.process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(nfsd.process.wait(), timeout=TERMINATE_TIMEOUT_SECONDS)
        except TimeoutError:
            nfsd.process.kill()
            await nfsd.process.wait()


def _locate_nfsd_binary() -> str:
    """Return the path to ``crb-nfsd``, preferring the daemon's own venv.

    Maturin's ``bindings = "bin"`` mode installs ``crb-nfsd`` into the
    venv's ``bin/`` directory alongside ``claude-remote-bash-daemon``.
    Looking adjacent to ``sys.executable`` is robust against PATH not
    including the venv (e.g., the daemon launched without venv activation).
    """
    adjacent = Path(sys.executable).parent / NFSD_BINARY
    if adjacent.is_file():
        return str(adjacent)

    on_path = shutil.which(NFSD_BINARY)
    if on_path is not None:
        return on_path

    raise DaemonError(
        f"Couldn't find `{NFSD_BINARY}` next to {sys.executable} or on PATH. "
        'Run `uv tool install --reinstall claude-remote-bash` (for installed daemons) '
        'or `uv sync --all-groups --all-packages` (workspace dev).'
    )


async def _read_listen_port(process: asyncio.subprocess.Process) -> int:
    """Parse ``LISTEN_PORT=N`` then ``READY`` from the child's stdout."""
    if process.stdout is None:
        raise DaemonError(f'{NFSD_BINARY} stdout is not piped — cannot read READY signal')
    port: int | None = None
    while True:
        line = await process.stdout.readline()
        if not line:
            raise DaemonError(f'{NFSD_BINARY} exited before signaling READY')
        text = line.decode('utf-8', errors='replace').rstrip('\n')
        if text.startswith('LISTEN_PORT='):
            port = int(text.removeprefix('LISTEN_PORT='))
        elif text == 'READY':
            if port is None:
                raise DaemonError(f'{NFSD_BINARY} signaled READY without LISTEN_PORT')
            return port
