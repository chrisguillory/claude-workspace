"""Empirical tests for ``claude_remote_bash.nfsd_manager.NfsdManager``.

These tests exercise spawn / refcount / shared-child / shutdown against a real
``crb-nfsd`` binary on PATH. They depend on ``uv sync --all-groups --all-packages``
having installed the binary into the venv's ``bin``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from claude_remote_bash.exceptions import DaemonError
from claude_remote_bash.nfsd_manager import NfsdManager


def _binary_available() -> bool:
    return shutil.which('crb-nfsd') is not None


skip_if_no_binary = pytest.mark.skipif(
    not _binary_available(),
    reason='crb-nfsd not on PATH (run `uv sync --all-groups --all-packages`)',
)


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Disposable served root with a file so subsequent NFS calls have something to find."""
    (tmp_path / 'hello.txt').write_text('hello\n')
    return tmp_path


@skip_if_no_binary
class TestNfsdManagerLifecycle:
    """Spawn / share / release / terminate end-to-end."""

    def test_acquire_spawns_a_child_and_reports_a_real_listen_port(self, tmp_root: Path) -> None:
        async def run() -> tuple[str, int, int]:
            mgr = NfsdManager()
            mount_id = await mgr.acquire(str(tmp_root), readonly=False)
            nfsd = mgr.get(mount_id)
            assert nfsd is not None
            pid = nfsd.process.pid
            port = nfsd.port
            await mgr.shutdown()
            return mount_id, pid, port

        mount_id, pid, port = asyncio.run(run())

        assert len(mount_id) == 32  # uuid.uuid4().hex
        assert pid > 0
        assert 1 <= port <= 65535
        # Port should be ephemeral, not 0
        assert port != 0

    def test_acquire_returns_distinct_mount_ids_for_distinct_keys(self, tmp_path: Path) -> None:
        """Two different roots → two different children, distinct mount_ids."""
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()

        async def run() -> tuple[str, str, int, int]:
            mgr = NfsdManager()
            id_a = await mgr.acquire(str(root_a), readonly=False)
            id_b = await mgr.acquire(str(root_b), readonly=False)
            nfsd_a = mgr.get(id_a)
            nfsd_b = mgr.get(id_b)
            assert nfsd_a is not None
            assert nfsd_b is not None
            pid_a = nfsd_a.process.pid
            pid_b = nfsd_b.process.pid
            await mgr.shutdown()
            return id_a, id_b, pid_a, pid_b

        id_a, id_b, pid_a, pid_b = asyncio.run(run())
        assert id_a != id_b
        assert pid_a != pid_b

    def test_two_acquires_same_key_share_the_child(self, tmp_root: Path) -> None:
        """Tab A and Tab B both mount the same `(root, readonly)` → one child, two mount_ids, refcount 2."""

        async def run() -> tuple[str, str, int, int]:
            mgr = NfsdManager()
            id_a = await mgr.acquire(str(tmp_root), readonly=False)
            id_b = await mgr.acquire(str(tmp_root), readonly=False)
            nfsd_a = mgr.get(id_a)
            nfsd_b = mgr.get(id_b)
            assert nfsd_a is not None
            assert nfsd_b is not None
            assert nfsd_a is nfsd_b  # shared NfsdProcess instance
            shared_pid = nfsd_a.process.pid
            refcount = nfsd_a.refcount
            await mgr.shutdown()
            return id_a, id_b, shared_pid, refcount

        id_a, id_b, shared_pid, refcount = asyncio.run(run())
        assert id_a != id_b
        assert shared_pid > 0
        assert refcount == 2

    def test_release_middle_holder_keeps_child_alive_for_remaining_holder(self, tmp_root: Path) -> None:
        """The scenario that justifies refcount: tab A unmounts, tab B keeps working."""

        async def run() -> tuple[bool, bool, bool]:
            mgr = NfsdManager()
            id_a = await mgr.acquire(str(tmp_root), readonly=False)
            id_b = await mgr.acquire(str(tmp_root), readonly=False)
            nfsd = mgr.get(id_a)
            assert nfsd is not None
            pid = nfsd.process.pid

            killed_first = await mgr.release(id_a)
            still_running_after_first_release = _pid_is_running(pid)

            killed_last = await mgr.release(id_b)
            still_running_after_last_release = _pid_is_running(pid)

            return (
                killed_first,
                still_running_after_first_release,
                (not still_running_after_last_release) and killed_last,
            )

        first_killed_child, alive_after_first, dead_after_last = asyncio.run(run())
        assert not first_killed_child, 'first release should not kill the shared child'
        assert alive_after_first, 'child should still be running after first release'
        assert dead_after_last, 'last release should terminate the child'

    def test_release_unknown_mount_id_is_idempotent_false(self) -> None:
        async def run() -> bool:
            mgr = NfsdManager()
            return await mgr.release('not-a-real-mount-id')

        assert asyncio.run(run()) is False

    def test_acquire_nonexistent_root_raises_daemon_error(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / 'does-not-exist'

        async def run() -> None:
            mgr = NfsdManager()
            await mgr.acquire(str(nonexistent), readonly=False)

        with pytest.raises(DaemonError, match='not an existing directory'):
            asyncio.run(run())

    def test_acquire_file_as_root_raises_daemon_error(self, tmp_path: Path) -> None:
        file_path = tmp_path / 'not-a-dir.txt'
        file_path.write_text('hi')

        async def run() -> None:
            mgr = NfsdManager()
            await mgr.acquire(str(file_path), readonly=False)

        with pytest.raises(DaemonError, match='not an existing directory'):
            asyncio.run(run())

    def test_shutdown_terminates_all_live_children(self, tmp_path: Path) -> None:
        root_a = tmp_path / 'a'
        root_b = tmp_path / 'b'
        root_a.mkdir()
        root_b.mkdir()

        async def run() -> tuple[int, int]:
            mgr = NfsdManager()
            id_a = await mgr.acquire(str(root_a), readonly=False)
            id_b = await mgr.acquire(str(root_b), readonly=False)
            nfsd_a = mgr.get(id_a)
            nfsd_b = mgr.get(id_b)
            assert nfsd_a is not None
            assert nfsd_b is not None
            pid_a = nfsd_a.process.pid
            pid_b = nfsd_b.process.pid
            await mgr.shutdown()
            return pid_a, pid_b

        pid_a, pid_b = asyncio.run(run())
        # Both children should be dead post-shutdown.
        assert not _pid_is_running(pid_a)
        assert not _pid_is_running(pid_b)


@skip_if_no_binary
class TestNfsdLoopbackPortIsReachable:
    """Confirms the spawned port is actually accepting TCP connections."""

    def test_loopback_port_accepts_tcp_connect(self, tmp_root: Path) -> None:
        async def run() -> int:
            mgr = NfsdManager()
            mount_id = await mgr.acquire(str(tmp_root), readonly=False)
            nfsd = mgr.get(mount_id)
            assert nfsd is not None
            port = nfsd.port

            # Plain TCP connect to confirm the listener is up. We don't speak
            # NFS RPC here — that's the tunnel-side integration test's job.
            _reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.close()
            await writer.wait_closed()

            await mgr.shutdown()
            return port

        port = asyncio.run(run())
        assert 1 <= port <= 65535


def _pid_is_running(pid: int) -> bool:
    """Return True iff sending signal 0 to ``pid`` succeeds (process exists)."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True
