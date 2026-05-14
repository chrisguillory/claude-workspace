"""End-to-end mux-protocol roundtrip: MountRequest → TunnelOpen → tunnel bytes → UnmountRequest.

Spawns a ``_Daemon`` in-process backed by a stubbed nfsd manager (TCP echo
server in place of a real ``crb-nfsd``) and exercises the full client-side
protocol surface. Kernel mount is NOT exercised — that needs a real binary
and real ``mount_nfs``, which is a manual cross-machine smoke.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from claude_remote_bash import auth as auth_module
from claude_remote_bash import paths as paths_module
from claude_remote_bash.auth import DaemonConfig
from claude_remote_bash.daemon import _Daemon
from claude_remote_bash.protocol import (
    read_frame,
    read_message,
    write_frame,
    write_message,
)
from claude_remote_bash.schemas.protocol import (
    AuthOk,
    AuthRequest,
    ErrorResponse,
    MountRequest,
    MountResponse,
    TunnelOk,
    TunnelOpen,
    UnmountRequest,
    UnmountResponse,
)


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the daemon config to a temp file with a known PSK."""
    cfg_path = tmp_path / 'config.json'
    monkeypatch.setattr(paths_module, 'DAEMON_CONFIG', cfg_path)
    monkeypatch.setattr(paths_module, 'DAEMON_CONFIG_LOCK', tmp_path / 'config.lock')
    monkeypatch.setattr(paths_module, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(auth_module, 'DAEMON_CONFIG', cfg_path)
    monkeypatch.setattr(auth_module, 'DAEMON_CONFIG_LOCK', tmp_path / 'config.lock')
    monkeypatch.setattr(auth_module, 'DATA_DIR', tmp_path)
    cfg = {
        'name': 'test-daemon',
        'auth_key': 'deadbeef' * 8,
        'shell': '/bin/zsh',
        'session_timeout_minutes': 1440,
    }
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


@pytest.fixture
async def running_daemon(isolated_config: Path) -> AsyncIterator[tuple[int, _EchoNfsdManager]]:
    """Boot a ``_Daemon`` with an echo-stub nfsd manager on 127.0.0.1:0."""
    _ = isolated_config  # required so monkeypatch is applied before _Daemon construction
    config = DaemonConfig(
        name='test-daemon',
        auth_key='deadbeef' * 8,
        shell='/bin/zsh',
        session_timeout_minutes=1440,
    )
    daemon = _Daemon(config)
    stub = _EchoNfsdManager()
    # The test substitutes a duck-typed echo stand-in for NfsdManager. The
    # daemon only calls .acquire/.release/.get/.shutdown — the stub matches
    # that surface but is not a subtype.
    daemon._nfsd_manager = stub  # type: ignore[assignment]  # duck-typed echo stub in place of NfsdManager
    server = await asyncio.start_server(daemon.handle_client, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port, stub
    finally:
        server.close()
        await server.wait_closed()
        await stub.shutdown()


@pytest.mark.asyncio
async def test_mount_returns_a_mount_id(running_daemon: tuple[int, _EchoNfsdManager]) -> None:
    port, _stub = running_daemon
    reader, writer = await _authed_conn(port)
    try:
        await write_message(writer, MountRequest(id='t1', root='/tmp', readonly=False))
        resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(resp, MountResponse), f'expected MountResponse, got {resp}'
        assert resp.mount_id.startswith('mid-')
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_tunnel_open_returns_a_channel_id_and_routes_bytes(
    running_daemon: tuple[int, _EchoNfsdManager],
) -> None:
    """Frames sent on the allocated channel reach the echo nfsd and come back."""
    port, _stub = running_daemon
    reader, writer = await _authed_conn(port)
    try:
        await write_message(writer, MountRequest(id='t2', root='/tmp', readonly=False))
        mount_resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(mount_resp, MountResponse)
        mount_id = mount_resp.mount_id

        await write_message(writer, TunnelOpen(mount_id=mount_id))
        tunnel_resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(tunnel_resp, TunnelOk)
        channel_id = tunnel_resp.channel_id
        assert channel_id == 1

        payload = b'hello-from-the-client'
        await write_frame(writer, channel_id, payload)

        ch, echoed = await asyncio.wait_for(read_frame(reader), timeout=3.0)
        assert ch == channel_id
        assert echoed == payload
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_two_tunnel_channels_get_distinct_ids_and_route_independently(
    running_daemon: tuple[int, _EchoNfsdManager],
) -> None:
    """The mux is the load-bearing piece — verify per-channel routing."""
    port, _stub = running_daemon
    reader, writer = await _authed_conn(port)
    try:
        await write_message(writer, MountRequest(id='t3', root='/tmp', readonly=False))
        mount_resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(mount_resp, MountResponse)
        mount_id = mount_resp.mount_id

        await write_message(writer, TunnelOpen(mount_id=mount_id))
        ok1 = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(ok1, TunnelOk)

        await write_message(writer, TunnelOpen(mount_id=mount_id))
        ok2 = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(ok2, TunnelOk)

        assert ok1.channel_id != ok2.channel_id

        # Interleave writes to both channels; expect each echo to come back on the right channel.
        await write_frame(writer, ok1.channel_id, b'channel-one')
        await write_frame(writer, ok2.channel_id, b'channel-two')

        echoes: dict[int, bytes] = {}
        for _ in range(2):
            ch, data = await asyncio.wait_for(read_frame(reader), timeout=3.0)
            echoes[ch] = data
        assert echoes[ok1.channel_id] == b'channel-one'
        assert echoes[ok2.channel_id] == b'channel-two'
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_unmount_request_releases_the_nfsd_and_reports_termination(
    running_daemon: tuple[int, _EchoNfsdManager],
) -> None:
    """Last holder unmount → child_terminated=True; nfsd registry is empty."""
    port, stub = running_daemon
    reader, writer = await _authed_conn(port)
    try:
        await write_message(writer, MountRequest(id='t4', root='/tmp', readonly=False))
        mount_resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(mount_resp, MountResponse)
        mount_id = mount_resp.mount_id
        assert stub.get(mount_id) is not None

        await write_message(writer, UnmountRequest(id='t4u', mount_id=mount_id))
        unmount_resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(unmount_resp, UnmountResponse)
        assert unmount_resp.mount_id == mount_id
        assert unmount_resp.child_terminated is True
        assert stub.get(mount_id) is None
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_tunnel_open_on_unknown_mount_id_errors(
    running_daemon: tuple[int, _EchoNfsdManager],
) -> None:
    port, _stub = running_daemon
    reader, writer = await _authed_conn(port)
    try:
        await write_message(writer, TunnelOpen(mount_id='not-a-real-id'))
        resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(resp, ErrorResponse)
        assert 'unknown mount_id' in resp.message
    finally:
        writer.close()
        await writer.wait_closed()


# -- Test fixtures / helpers ---------------------------------------------------


@dataclass
class _StubNfsd:
    """Shape-matches ``NfsdProcess.port`` so ``_Daemon._open_tunnel`` works."""

    port: int
    refcount: int = 1


class _EchoNfsdManager:
    """In-process echo-server stand-in for ``NfsdManager``.

    Each ``acquire`` starts a TCP echo server on 127.0.0.1:0 and registers
    it as a stub nfsd. ``release`` shuts the server down. ``get`` returns
    the ``port`` attribute the daemon needs to open its tunnel-side socket.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, tuple[_StubNfsd, asyncio.Server]] = {}
        self._counter = 0

    async def acquire(self, root: str, readonly: bool) -> str:
        _ = root, readonly  # echo stub ignores them

        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        return
                    writer.write(chunk)
                    await writer.drain()
            finally:
                writer.close()

        server = await asyncio.start_server(echo, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        self._counter += 1
        mount_id = f'mid-{self._counter:08x}'
        self._by_id[mount_id] = (_StubNfsd(port=port), server)
        return mount_id

    async def release(self, mount_id: str) -> bool:
        entry = self._by_id.pop(mount_id, None)
        if entry is None:
            return False
        _stub, server = entry
        server.close()
        await server.wait_closed()
        return True

    def get(self, mount_id: str) -> _StubNfsd | None:
        entry = self._by_id.get(mount_id)
        return entry[0] if entry else None

    async def shutdown(self) -> None:
        for _stub, server in list(self._by_id.values()):
            server.close()
            await server.wait_closed()
        self._by_id.clear()


async def _authed_conn(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to the test daemon and complete the PSK handshake."""
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    await write_message(writer, AuthRequest(key='deadbeef' * 8))
    resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
    assert isinstance(resp, AuthOk), f'expected AuthOk, got {resp}'
    return reader, writer
