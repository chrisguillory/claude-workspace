from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from claude_remote_bash.exceptions import DaemonError

__all__ = [
    'NfsdManager',
    'NfsdProcess',
]

logger = logging.getLogger(__name__)

NFSD_BINARY = 'crb-nfsd'
"""Expected on PATH after `uv sync --all-groups --all-packages`."""

READY_TIMEOUT_SECONDS = 10.0
"""How long to wait for the spawned binary to print `LISTEN_PORT=N` + `READY`."""

TERMINATE_TIMEOUT_SECONDS = 3.0
"""Grace window between SIGTERM and SIGKILL during shutdown."""


@dataclass
class NfsdProcess:
    """A running ``crb-nfsd`` child serving a single ``(root, readonly)``."""

    mount_id: str
    """Opaque identifier returned to clients; the daemon's tunnel handler uses it to look up the port."""

    root: Path
    """Canonical absolute path the child is serving."""

    readonly: bool

    port: int
    """The loopback port the child is listening on (printed via LISTEN_PORT=N)."""

    process: asyncio.subprocess.Process

    refcount: int = 1
    """Number of clients holding this mount. SIGTERM on zero."""

    exit_watcher: asyncio.Task[None] | None = field(default=None, repr=False)
    """Background task that observes unexpected child death and evicts from registry."""


class NfsdManager:
    """Tracks active ``crb-nfsd`` children with refcounted lifecycle.

    A single child serves all clients that share the same ``(root, readonly)``
    tuple. ``acquire`` returns the existing child if one matches, spawning
    only when no match exists. ``release`` drops the refcount; the child is
    SIGTERMed when the count reaches zero.

    Child death between acquire and release is fail-stop: the entry is
    evicted from the registry, and ``get(mount_id)`` returns ``None``.
    Respawning would create a fresh fileid sequence while the kernel client
    still caches fileids from the previous instance — a recipe for silent
    data corruption where an Edit lands in a different file than expected.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[Path, bool], NfsdProcess] = {}
        self._by_id: dict[str, NfsdProcess] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, root: str, readonly: bool) -> NfsdProcess:
        """Get or spawn an nfsd for ``(canonical(root), readonly)``.

        Increments the refcount and returns the shared NfsdProcess. The
        daemon side passes ``--block-prefix $HOME/.crb/`` so the served tree
        can't loop back through the mount-supervisor's own state directory.
        """
        canonical = Path(root).expanduser().resolve()
        if not canonical.is_dir():
            raise DaemonError(f'root is not an existing directory: {canonical}')

        async with self._lock:
            key = (canonical, readonly)
            existing = self._by_key.get(key)
            if existing is not None:
                existing.refcount += 1
                logger.info(
                    'Reusing nfsd: root=%s readonly=%s mount_id=%s refcount=%d',
                    canonical,
                    readonly,
                    existing.mount_id,
                    existing.refcount,
                )
                return existing

            nfsd = await self._spawn(canonical, readonly)
            self._by_key[key] = nfsd
            self._by_id[nfsd.mount_id] = nfsd
            logger.info(
                'Spawned nfsd: root=%s readonly=%s mount_id=%s port=%d',
                canonical,
                readonly,
                nfsd.mount_id,
                nfsd.port,
            )
            return nfsd

    async def release(self, mount_id: str) -> bool:
        """Drop one hold on ``mount_id``. Returns True iff this released the last holder.

        Idempotent — releasing an unknown or already-dead mount_id returns False.
        """
        async with self._lock:
            nfsd = self._by_id.get(mount_id)
            if nfsd is None:
                return False

            nfsd.refcount -= 1
            if nfsd.refcount > 0:
                logger.info('Released nfsd hold: mount_id=%s refcount=%d', mount_id, nfsd.refcount)
                return False

            await self._terminate(nfsd)
            self._by_key.pop((nfsd.root, nfsd.readonly), None)
            self._by_id.pop(mount_id, None)
            logger.info('Terminated nfsd: mount_id=%s', mount_id)
            return True

    def get(self, mount_id: str) -> NfsdProcess | None:
        """Look up the running nfsd by mount_id, or None if unknown/dead."""
        return self._by_id.get(mount_id)

    async def shutdown(self) -> None:
        """SIGTERM all children and await their exit. Used by daemon shutdown."""
        async with self._lock:
            procs = list(self._by_id.values())
            self._by_id.clear()
            self._by_key.clear()
        for nfsd in procs:
            await self._terminate(nfsd)
        if procs:
            logger.info('Shut down %d nfsd child(ren)', len(procs))

    async def _spawn(self, root: Path, readonly: bool) -> NfsdProcess:
        binary = shutil.which(NFSD_BINARY)
        if binary is None:
            raise DaemonError(
                f"Couldn't find `{NFSD_BINARY}` on PATH. "
                'Run `uv sync --all-groups --all-packages` from the workspace root.'
            )

        # $HOME/.crb/ holds mount registry and per-peer mountpoints. Block-prefixing
        # it stops the served tree from confusing the supervisor by exposing its
        # own state directory through the mount.
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

        mount_id = uuid.uuid4().hex
        nfsd = NfsdProcess(
            mount_id=mount_id,
            root=root,
            readonly=readonly,
            port=port,
            process=process,
        )
        nfsd.exit_watcher = asyncio.create_task(self._watch_exit(nfsd))
        return nfsd

    async def _watch_exit(self, nfsd: NfsdProcess) -> None:
        """Observe unexpected child death; evict from registry so the mount is fail-stop.

        Cancellation (during ``_terminate``) propagates as ``CancelledError``
        before the eviction block runs — that's the intended path when we're
        the ones tearing down. The post-wait code only runs on natural death.
        """
        await nfsd.process.wait()
        async with self._lock:
            if self._by_id.pop(nfsd.mount_id, None) is not None:
                self._by_key.pop((nfsd.root, nfsd.readonly), None)
                logger.warning(
                    'nfsd child pid=%d root=%s died unexpectedly with code %d; '
                    'mount_id=%s now invalidated — tunnel claims will fail.',
                    nfsd.process.pid,
                    nfsd.root,
                    nfsd.process.returncode,
                    nfsd.mount_id,
                )

    async def _terminate(self, nfsd: NfsdProcess) -> None:
        """Cancel the exit watcher, SIGTERM the child, SIGKILL if it doesn't exit in time."""
        if nfsd.exit_watcher is not None and not nfsd.exit_watcher.done():
            nfsd.exit_watcher.cancel()
            # asyncio.wait awaits the cancelled task without re-raising at this
            # frame — we initiated the cancellation, so the cancellation state
            # is not a signal we want to propagate up.
            await asyncio.wait([nfsd.exit_watcher])
        if nfsd.process.returncode is not None:
            return  # already dead
        try:
            nfsd.process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(nfsd.process.wait(), timeout=TERMINATE_TIMEOUT_SECONDS)
        except TimeoutError:
            nfsd.process.kill()
            await nfsd.process.wait()


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
