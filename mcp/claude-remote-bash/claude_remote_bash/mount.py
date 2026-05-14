"""Client-side mount supervisor: discover, MountRequest, local listener, mount_nfs, block, umount."""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path

from claude_remote_bash.client import authenticate, open_connection_any
from claude_remote_bash.exceptions import (
    DaemonError,
    HostNotFoundError,
    RemoteBashError,
)
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
]

logger = logging.getLogger(__name__)

MOUNT_NFS_FLAGS = (
    'vers=3,tcp,nolocks,actimeo=0,rdirplus,rsize=65536,wsize=65536,soft,timeo=50,retrans=2,deadtimeout=30,intr'
)
"""Empirically-validated `mount_nfs(8)` flags. `actimeo=0,noac` keeps Edit-tool
read-after-write coherent; `soft+timeo+retrans+deadtimeout` bound recovery to
~30s on tunnel failure rather than indefinite kernel hangs."""


class MountError(RemoteBashError):
    """Failure during the mount flow — mount_nfs error, post-mount verify failed, etc."""

    prefix = 'Mount error'


def default_mountpoint(peer_alias: str, remote_path: str) -> Path:
    """``crb mount m2:~/projects/foo`` → ``~/.crb/host/m2/foo/``."""
    basename = Path(remote_path).name or 'root'
    return MOUNTS_ROOT / peer_alias / basename


async def crb_mount_blocking(
    *,
    peer_alias: str,
    remote_path: str,
    mountpoint: Path,
    readonly: bool,
    ips: Sequence[str] | None = None,
    port: int | None = None,
) -> None:
    """Mount ``peer_alias:remote_path`` at ``mountpoint``. Blocks until SIGINT/SIGTERM.

    On signal: sends ``UnmountRequest`` to the peer, runs ``umount`` locally,
    closes the control connection.

    ``ips``/``port`` override mDNS discovery (used by tests). When omitted,
    discovers via the existing browse-cache flow.
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

        local_listener = await _start_local_listener(peer_alias, mount_id, ips, port)
        local_port = local_listener.sockets[0].getsockname()[1]
        logger.info('Local NFS bridge listening on 127.0.0.1:%d', local_port)

        await _run_mount_nfs(local_port, mountpoint)
        _post_mount_macos_tweaks(mountpoint)
        logger.info('Mounted %s:%s at %s', peer_alias, remote_path, mountpoint)
        print(f'mounted {peer_alias}:{remote_path} at {mountpoint}')
        print('Press Ctrl-C to unmount.')

        await _wait_for_signal()

        logger.info('Unmounting %s', mountpoint)
        await _teardown(mountpoint, writer, mount_id, local_listener)
    finally:
        writer.close()
        await writer.wait_closed()


async def _resolve(peer_alias: str) -> tuple[Sequence[str], int]:
    """Find ``peer_alias`` in the mDNS cache (fresh-browses on cache miss)."""
    cache = await browse_and_cache()
    hit = lookup_alias(cache, peer_alias)
    if hit is None:
        cache = await browse_fresh_and_cache()
        hit = lookup_alias(cache, peer_alias)
    if hit is None:
        raise_host_not_found(peer_alias)
        raise HostNotFoundError(peer_alias)  # unreachable but satisfies the type checker
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
    peer_alias: str,
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

        await asyncio.gather(kernel_to_peer(), peer_to_kernel(), return_exceptions=True)
    finally:
        peer_writer.close()


async def _run_mount_nfs(local_port: int, mountpoint: Path) -> None:
    """Invoke ``mount_nfs`` and verify the kernel actually mounted."""
    flags = f'{MOUNT_NFS_FLAGS},port={local_port},mountport={local_port}'
    args = ['/sbin/mount_nfs', '-o', flags, '127.0.0.1:/', str(mountpoint)]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise MountError(f'mount_nfs failed (rc={proc.returncode}): {stderr.decode("utf-8", errors="replace").strip()}')

    mount_table = await _read_mount_table()
    if str(mountpoint) not in mount_table:
        raise MountError(f'mount_nfs returned 0 but {mountpoint} is not in the kernel mount table')


async def _read_mount_table() -> str:
    proc = await asyncio.create_subprocess_exec(
        '/sbin/mount',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode('utf-8', errors='replace')


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


async def _teardown(
    mountpoint: Path,
    peer_writer: asyncio.StreamWriter,
    mount_id: str,
    local_listener: asyncio.Server,
) -> None:
    """Force-umount the mountpoint, tell the peer to release, close the listener."""
    proc = await asyncio.create_subprocess_exec(
        '/sbin/umount',
        '-f',
        str(mountpoint),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning('umount returned %d: %s', proc.returncode, stderr.decode().strip())

    try:
        await write_message(peer_writer, UnmountRequest(id='cli-unmount-1', mount_id=mount_id))
    except (ConnectionError, BrokenPipeError):
        pass  # peer connection already broken — nothing actionable

    local_listener.close()
    await local_listener.wait_closed()
