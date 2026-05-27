"""Client-side mount supervisor: discover, MountRequest, local listener, mount_nfs, block, umount."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from claude_remote_bash.client import authenticate, open_connection_any
from claude_remote_bash.exceptions import (
    DaemonError,
    HostNotFoundError,
    RemoteBashError,
)
from claude_remote_bash.mounts_registry import find_by_mountpoint, remove_entry
from claude_remote_bash.paths import MOUNTS_ROOT
from claude_remote_bash.protocol import (
    read_frame,
    read_message,
    write_frame,
    write_message,
)
from claude_remote_bash.resolve import (
    browse_and_cache,
    browse_fresh_and_cache,
    lookup_alias,
    raise_host_not_found,
)
from claude_remote_bash.schemas.protocol import (
    ErrorResponse,
    MountRequest,
    MountResponse,
    TunnelOk,
    TunnelOpen,
    UnmountRequest,
)

__all__ = [
    'MountError',
    'crb_mount_blocking',
    'default_mountpoint',
    'spawn_detached_supervisor',
    'supervisor_alive',
    'terminate_supervisor',
]

SUPERVISOR_READY_TIMEOUT_SECONDS = 30.0
"""Bound on how long ``crb mount`` waits for the spawned supervisor to print ``READY``."""

SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS = 30.0
"""Bound on how long ``crb umount`` waits for the supervisor to remove its registry entry after SIGTERM."""

MOUNT_VERIFY_TIMEOUT_SECONDS = 10.0
"""Maximum time to wait for the kernel mount table to reflect a successful ``mount_nfs``.

``mount_nfs`` returns 0 once it has handed the mount off to the kernel, but
the mount becomes observable via ``lstat`` (and thus ``Path.is_mount()``)
only after the NFS handshake + tunnel setup over the local bridge complete
— empirically ~3 seconds on a healthy LAN. The poll loop in
``_run_mount_nfs`` retries until the mount is visible or this deadline
elapses; the latter indicates a true failure (peer unreachable, etc.)."""

MOUNT_VERIFY_POLL_INTERVAL_SECONDS = 0.25
"""Polling interval for the post-``mount_nfs`` verification loop.

Each iteration spawns ``/sbin/mount`` — too-frequent polling adds subprocess
pressure that empirically slows ``mount_nfs``'s own kernel registration."""

logger = logging.getLogger(__name__)

MOUNT_NFS_FLAGS = (
    'vers=3,tcp,nolocks,actimeo=0,rdirplus,rsize=65536,wsize=65536,soft,timeo=50,retrans=2,deadtimeout=30,intr'
)
"""Always-applied `mount_nfs(8)` flags. `actimeo=0,noac` keeps Edit-tool
read-after-write coherent; `soft+timeo+retrans+deadtimeout` bound recovery to
~30s on tunnel failure rather than indefinite kernel hangs. The ``nobrowse``
flag is appended per-mount unless ``browse=True`` is set — see ``_run_mount_nfs``."""


class MountError(RemoteBashError):
    """Failure during the mount flow — mount_nfs error, post-mount verify failed, etc."""

    prefix = 'Mount error'


def default_mountpoint(peer_alias: str, remote_path: str) -> Path:
    """``crb mount m2:~/projects/foo`` → ``~/.crb/host/m2/foo/``."""
    basename = Path(remote_path).name or 'root'
    return MOUNTS_ROOT / peer_alias / basename


def spawn_detached_supervisor(
    *,
    peer_alias: str,
    remote_path: str,
    mountpoint: Path,
    readonly: bool,
    browse: bool = False,
) -> tuple[int, str]:
    """Fork a detached supervisor and wait for it to signal ``READY``.

    Returns ``(supervisor_pid, mount_id)``. Raises ``MountError`` if the
    supervisor exits before READY or fails to signal within
    ``SUPERVISOR_READY_TIMEOUT_SECONDS``. On failure, stderr from the
    supervisor (formatted via its ErrorBoundary) is included verbatim.
    """
    args = [
        sys.executable,
        '-m',
        'claude_remote_bash.mount_supervisor',
        peer_alias,
        remote_path,
        str(mountpoint),
    ]
    if readonly:
        args.append('--readonly')
    if browse:
        args.append('--browse')

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )

    try:
        line = _read_first_line_with_timeout(proc.stdout, SUPERVISOR_READY_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        proc.terminate()
        proc.wait(timeout=5.0)
        raise MountError(f'Supervisor did not signal READY within {SUPERVISOR_READY_TIMEOUT_SECONDS:.0f}s') from exc

    if line is None:
        # Supervisor exited before signaling — its ErrorBoundary handler wrote the reason to stderr.
        proc.wait(timeout=5.0)
        stderr = proc.stderr.read() if proc.stderr else ''
        raise MountError(f'Supervisor exited (rc={proc.returncode}): {stderr.strip()}')

    line = line.rstrip('\n')
    if not line.startswith('READY '):
        proc.terminate()
        proc.wait(timeout=5.0)
        raise MountError(f'Unexpected supervisor handshake: {line!r}')

    mount_id = line.removeprefix('READY ').strip()
    return proc.pid, mount_id


def supervisor_alive(pid: int) -> bool:
    """Return True iff the supervisor process is still running."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def terminate_supervisor(mountpoint: Path) -> None:
    """SIGTERM the supervisor for ``mountpoint`` and wait for it to clean up.

    "Clean up" = supervisor removes its own registry entry, which it does
    in the ``finally`` block of ``mount_supervisor._supervise``. Polling
    the registry rather than ``waitpid`` is necessary because the caller
    is not the supervisor's parent.
    """
    abs_mp = mountpoint.resolve()
    entry = find_by_mountpoint(abs_mp)
    if entry is None:
        raise MountError(f'No active mount at {abs_mp}')

    try:
        os.kill(entry.supervisor_pid, signal.SIGTERM)
    except ProcessLookupError as exc:
        # Stale registry entry — supervisor died without cleaning up.
        remove_entry(abs_mp)
        raise MountError(
            f'Supervisor pid={entry.supervisor_pid} was already gone — removed stale registry entry.'
        ) from exc

    deadline = time.monotonic() + SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if find_by_mountpoint(abs_mp) is None:
            return
        time.sleep(0.2)

    raise MountError(
        f'Supervisor pid={entry.supervisor_pid} did not unwind within {SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS:.0f}s.'
    )


async def crb_mount_blocking(
    *,
    peer_alias: str,
    remote_path: str,
    mountpoint: Path,
    readonly: bool,
    browse: bool = False,
    ips: Sequence[str] | None = None,
    port: int | None = None,
    on_mounted: Callable[[str], None] | None = None,
) -> None:
    """Mount ``peer_alias:remote_path`` at ``mountpoint``. Blocks until SIGINT/SIGTERM.

    On signal: sends ``UnmountRequest`` to the peer, runs ``umount`` locally,
    closes the control connection.

    ``ips``/``port`` override mDNS discovery (used by tests). When omitted,
    discovers via the existing browse-cache flow.

    ``browse=True`` makes the mount appear in Finder's "Computer" view and
    sidebar. Default is hidden (``nobrowse`` mount flag) — see
    ``MOUNT_NFS_FLAGS``.

    ``on_mounted`` fires once with the peer-assigned ``mount_id`` right
    after ``mount_nfs`` succeeds — supervisors use it to write a registry
    entry and signal ``READY`` to the spawning parent.
    """
    if ips is None or port is None:
        ips, port = await _resolve(peer_alias)

    mountpoint.mkdir(parents=True, exist_ok=True)
    if any(mountpoint.iterdir()):
        raise MountError(f'mountpoint not empty: {mountpoint}')

    reader, writer = await open_connection_any(ips, port)
    try:
        await authenticate(reader, writer)
        mount_id = await _request_mount(reader, writer, remote_path, readonly)
        logger.info('Peer %s spawned nfsd: mount_id=%s', peer_alias, mount_id[:8])

        local_listener = await _start_local_listener(mount_id, ips, port)
        async with local_listener:
            local_port = local_listener.sockets[0].getsockname()[1]
            logger.info('Local NFS bridge listening on 127.0.0.1:%d', local_port)

            await _run_mount_nfs(local_port, mountpoint, browse=browse)
            _post_mount_macos_tweaks(mountpoint)
            logger.info('Mounted %s:%s at %s', peer_alias, remote_path, mountpoint)

            if on_mounted is not None:
                on_mounted(mount_id)

            await _wait_for_signal()

            logger.info('Unmounting %s', mountpoint)
            await _release_mount(mountpoint, writer, mount_id)
    finally:
        writer.close()
        await writer.wait_closed()


async def _resolve(peer_alias: str) -> tuple[Sequence[str], int]:
    """Find ``peer_alias`` in the mDNS cache and enforce canonical-case match.

    Aliases are advertised via mDNS in a specific canonical case (whatever the
    target set via ``set-name``). ``crb discover`` prints them verbatim. Mount
    is strict about case: typing ``m2`` when the alias is ``M2`` is user error
    we surface, not paper over — silently normalizing leads to mountpoint paths
    that don't match the kernel's APFS-canonical case, which then break
    verification, registry lookup, and umount in ways that look like unrelated
    bugs (this PR's original symptom).

    Lookup-the-cache is case-insensitive (ergonomic typo tolerance for finding
    the host); the post-lookup canonical check is what gates mounting.
    """
    cache = await browse_and_cache()
    hit = lookup_alias(cache, peer_alias)
    if hit is None:
        cache = await browse_fresh_and_cache()
        hit = lookup_alias(cache, peer_alias)
    if hit is None:
        raise_host_not_found(peer_alias)
        raise HostNotFoundError(peer_alias)  # unreachable but satisfies the type checker

    canonical = next(
        (e.alias for e in cache.all_hosts() if e.alias.lower() == peer_alias.lower()),
        None,
    )
    if canonical is not None and canonical != peer_alias:
        raise MountError(
            f"Alias '{peer_alias}' is not canonical. Use '{canonical}' "
            f'(aliases are case-sensitive at mount time; run `crb discover` for canonical names).'
        )

    ips, port = hit
    return list(ips), port


async def _request_mount(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    remote_path: str,
    readonly: bool,
) -> str:
    """Send MountRequest; return mount_id from MountResponse."""
    await write_message(writer, MountRequest(id='cli-mount-1', root=remote_path, readonly=readonly))
    resp = await read_message(reader)
    if isinstance(resp, ErrorResponse):
        raise DaemonError(resp.message)
    if not isinstance(resp, MountResponse):
        raise DaemonError(f'unexpected response: {resp.type}')
    return resp.mount_id


async def _start_local_listener(
    mount_id: str,
    peer_ips: Sequence[str],
    peer_port: int,
) -> asyncio.Server:
    """Bind 127.0.0.1:0 and bridge each accept into a fresh tunnel channel."""

    async def handle_kernel_conn(kernel_reader: asyncio.StreamReader, kernel_writer: asyncio.StreamWriter) -> None:
        try:
            await _bridge_kernel_conn_to_peer(kernel_reader, kernel_writer, mount_id, peer_ips, peer_port)
        finally:
            kernel_writer.close()
            await kernel_writer.wait_closed()

    return await asyncio.start_server(handle_kernel_conn, '127.0.0.1', 0)


async def _bridge_kernel_conn_to_peer(
    kernel_reader: asyncio.StreamReader,
    kernel_writer: asyncio.StreamWriter,
    mount_id: str,
    peer_ips: Sequence[str],
    peer_port: int,
) -> None:
    """One kernel-client TCP conn ↔ one fresh PSK-authed peer-conn + tunnel channel.

    No M3-side multiplexing: each kernel connection opens its own peer-conn,
    authenticates, claims a tunnel channel. Adds PSK-auth overhead per conn
    but keeps the M3 side simple. The daemon's mux machinery is forward-
    compatible — adding M3 mux later is additive.
    """
    peer_reader, peer_writer = await open_connection_any(peer_ips, peer_port)
    try:
        await authenticate(peer_reader, peer_writer)
        await write_message(peer_writer, TunnelOpen(mount_id=mount_id))
        resp = await read_message(peer_reader)
        if not isinstance(resp, TunnelOk):
            raise DaemonError(f'TunnelOpen rejected: {resp}')
        channel_id = resp.channel_id

        async def kernel_to_peer() -> None:
            while True:
                chunk = await kernel_reader.read(65536)
                await write_frame(peer_writer, channel_id, chunk)
                if not chunk:
                    return

        async def peer_to_kernel() -> None:
            while True:
                ch, payload = await read_frame(peer_reader)
                if ch != channel_id:
                    continue  # control-channel chatter is unexpected here; ignore
                if not payload:
                    return
                kernel_writer.write(payload)
                await kernel_writer.drain()

        # FIRST_COMPLETED: as soon as one direction terminates (kernel EOF on
        # unmount, or peer-side zero-frame on nfsd shutdown), tear down the
        # other side. With ``gather`` we'd hang forever when the daemon
        # cancels its pump in response to our zero-frame and consequently
        # never sends one back — leaving our ``peer_to_kernel`` blocked.
        k2p = asyncio.create_task(kernel_to_peer(), name=f'k2p-ch{channel_id}')
        p2k = asyncio.create_task(peer_to_kernel(), name=f'p2k-ch{channel_id}')
        await asyncio.wait({k2p, p2k}, return_when=asyncio.FIRST_COMPLETED)
        for task in (k2p, p2k):
            if not task.done():
                task.cancel()
        await asyncio.wait({k2p, p2k})
        for task in (k2p, p2k):
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    'bridge %s exited with %s: %s',
                    task.get_name(),
                    type(exc).__name__,
                    exc,
                    exc_info=exc,
                )
    finally:
        peer_writer.close()


async def _run_mount_nfs(local_port: int, mountpoint: Path, *, browse: bool) -> None:
    """Invoke ``mount_nfs`` and verify the kernel actually mounted."""
    flags = MOUNT_NFS_FLAGS
    if not browse:
        flags = f'{flags},nobrowse'
    flags = f'{flags},port={local_port},mountport={local_port}'
    args = ['/sbin/mount_nfs', '-o', flags, '127.0.0.1:/', str(mountpoint)]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise MountError(f'mount_nfs failed (rc={proc.returncode}): {stderr.decode("utf-8", errors="replace").strip()}')

    # Poll the kernel mount table until the mount becomes visible or we
    # time out. ``mount_nfs`` returns 0 once it has handed the mount off
    # to the kernel, but the kernel finalizes registration asynchronously
    # (NFS handshake + tunnel setup over the local bridge) — empirically
    # ~3 seconds on a healthy LAN.
    #
    # We use ``/sbin/mount`` (which reads via ``getfsstat``) rather than
    # ``Path.is_mount()`` because ``lstat`` on the NFS mountpoint blocks
    # waiting for a GETATTR RPC; that RPC traverses our local bridge,
    # which can't run handlers while this coroutine is blocked in
    # ``lstat`` — self-deadlock that resolves only at the NFS soft timeout.
    deadline = asyncio.get_running_loop().time() + MOUNT_VERIFY_TIMEOUT_SECONDS
    while not await _kernel_has_mount_at(mountpoint):
        if asyncio.get_running_loop().time() >= deadline:
            # Defensive umount — mount_nfs may have a live entry the registry
            # can't see. Without this, the failure leaves a zombie. Result is
            # logged unconditionally for diagnostics; the MountError below is
            # the user-facing failure regardless of umount outcome.
            rc, stderr_text = await _run_umount(mountpoint)
            logger.info('defensive umount rc=%s stderr=%s', rc, stderr_text)
            raise MountError(f'mount_nfs returned 0 but {mountpoint} is not a kernel mount point')
        await asyncio.sleep(MOUNT_VERIFY_POLL_INTERVAL_SECONDS)


async def _kernel_has_mount_at(mountpoint: Path) -> bool:
    """True iff ``/sbin/mount`` lists an entry with destination matching ``mountpoint``.

    Exact string match — ``_resolve`` upstream enforces canonical-case alias, so
    ``mountpoint`` always contains the canonical case the kernel mount table records.

    Uses ``/sbin/mount`` (reads ``getfsstat``) rather than ``Path.is_mount()`` because
    ``lstat`` on an NFS mountpoint blocks on a GETATTR RPC that traverses this
    process's own local bridge — self-deadlock while we hold the event loop.
    """
    proc = await asyncio.create_subprocess_exec(
        '/sbin/mount',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    target = str(mountpoint)
    return any(_parse_mount_dest(line) == target for line in stdout.decode('utf-8', errors='replace').splitlines())


def _parse_mount_dest(line: str) -> str | None:
    """Extract the destination path from one ``/sbin/mount`` output line, or ``None`` if unparseable.

    macOS ``mount(8)`` format: ``<src> on <dest> (<opts>)``. The src has no
    spaces (always ``host:/path`` for NFS, ``/dev/...`` for block devices);
    the dest can contain spaces, parens, and even the literal ``" on "``.
    ``partition`` finds the first ``" on "`` (separating src from rest);
    ``rpartition`` finds the trailing ``" ("`` (anchoring the opts blob).
    """
    _, _, after_on = line.partition(' on ')
    if not after_on:
        return None
    dest, sep, _opts = after_on.rpartition(' (')
    return dest if sep else None


async def _run_umount(mountpoint: Path) -> tuple[int | None, str]:
    """Run ``umount -f <mountpoint>``. Returns ``(returncode, decoded-stderr)``.

    Lower-layer primitive. Callers decide how to log non-zero exits. ``returncode``
    is ``int | None`` to match asyncio's type — after ``await proc.communicate()``
    the process is terminated and returncode is in practice always ``int``, but
    mypy strict requires the honest type.
    """
    proc = await asyncio.create_subprocess_exec(
        '/sbin/umount',
        '-f',
        str(mountpoint),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    return proc.returncode, stderr.decode().strip()


def _post_mount_macos_tweaks(mountpoint: Path) -> None:
    """Tell Spotlight and Time Machine to leave the mount alone.

    Without these, Spotlight tries to index the mount on first Finder open
    (1000s of RPCs/sec) and Time Machine eventually walks the entire tree
    into the next backup. Both make the mount feel terrible.
    """
    subprocess.run(['/usr/bin/mdutil', '-i', 'off', str(mountpoint)], check=False, capture_output=True)
    subprocess.run(['/usr/bin/tmutil', 'addexclusion', str(mountpoint)], check=False, capture_output=True)


async def _wait_for_signal() -> None:
    """Block until SIGINT or SIGTERM."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()


async def _release_mount(
    mountpoint: Path,
    peer_writer: asyncio.StreamWriter,
    mount_id: str,
) -> None:
    """Client-side release: umount the local kernel mount and send ``UnmountRequest`` to the peer.

    Distinct from peer-side ``NfsdManager.release(mount_id)`` (which is invoked by the
    daemon in response to our ``UnmountRequest`` and decrements the peer's refcount).
    The local listener is owned by ``crb_mount_blocking``'s ``async with``; not closed here.
    """
    rc, stderr_text = await _run_umount(mountpoint)
    if rc != 0:
        logger.warning('umount returned %s: %s', rc, stderr_text)

    try:
        await write_message(peer_writer, UnmountRequest(id='cli-unmount-1', mount_id=mount_id))
    except (ConnectionError, BrokenPipeError):
        pass  # peer connection already broken — nothing actionable


def _read_first_line_with_timeout(stream: object, timeout: float) -> str | None:
    """Read one line from ``stream`` with timeout. Returns None on EOF.

    Uses a daemon thread because Popen pipe streams are blocking and the
    selectors module only signals readiness, not full-line availability.
    """
    result: queue.Queue[str | None] = queue.Queue(maxsize=1)

    def reader() -> None:
        line = stream.readline() if stream is not None else ''  # type: ignore[attr-defined]  # Popen stdout is IO[str] in text mode
        result.put(line if line else None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        return result.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError from exc
