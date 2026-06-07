"""Tests for cc_lib.mcp.server_registry -- McpServerInfo, register (CM), read_all."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
from cc_lib import os_process
from cc_lib.mcp.server_registry import McpServerInfo, find_one, read_all, register
from cc_lib.types import CCVersion


class TestRegistry:
    """Registry edges + regression-bait that empirical adoption won't catch."""

    async def test_empty_session(self) -> None:
        assert list(read_all('no-such-session')) == []

    async def test_round_trip_preserves_info(self, info: McpServerInfo) -> None:
        """Write -> read back -> identical wire form. Schema-change regression bait.

        Asserts wire fidelity (same JSON), not Python-object equality — container
        types may differ across the boundary (e.g. tuple in, list out); production
        readers only access fields.
        """
        async with register(info):
            (entry,) = read_all(info.session_id)
            assert entry.model_dump_json() == info.model_dump_json()

    async def test_skips_dead_pid(self, info: McpServerInfo, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dispositive read_all behavior — without this, stale entries leak."""
        async with register(info):
            monkeypatch.setattr('cc_lib.os_process.is_alive', lambda _pid: False)
            assert list(read_all(info.session_id)) == []

    async def test_skips_recycled_pid(self, info: McpServerInfo) -> None:
        """Recycle-safety: a live PID whose create_time drifted from the anchor reads as dead.

        Hard to stage live (real PID reuse is near-impossible to force), so it earns a
        regression test. With bare pid_exists this stale entry would survive; the
        ProcessHandle create_time match is what filters it.
        """
        recycled = info.__replace__(created_at=info.created_at - timedelta(hours=1))
        async with register(recycled):
            assert list(read_all(recycled.session_id)) == []

    async def test_register_cleans_up_on_exit(self, info: McpServerInfo, workspace_dir: Path) -> None:
        """File written on enter, removed on exit. The CM contract."""
        path = workspace_dir / 'mcp' / 'registry' / info.session_id / f'{info.name}-{info.mcp_pid}.json'
        async with register(info):
            assert path.is_file()
        assert not path.exists()

    async def test_find_one_tolerates_concurrent_deregister(
        self, info: McpServerInfo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sibling unlinked between glob and read is skipped, not fatal.

        The registry is lock-free: ``register``'s ``finally`` unlinks an entry
        while a concurrent ``find_one`` is mid-scan. Hard to stage live (a
        microsecond glob->read window), so it earns a regression test. Without
        the FileNotFoundError skip, find_one crashes on the vanished lower-pid
        sibling before reaching the live target.
        """
        # Lower-pid sibling sorts first and vanishes mid-scan; current-pid entry is the live target.
        vanishing = info.__replace__(mcp_pid=1, sock_path='/tmp/dead.sock')
        vanishing_file = f'{vanishing.name}-{vanishing.mcp_pid}.json'
        real_read = Path.read_bytes

        def racing_read(self: Path) -> bytes:
            if self.name == vanishing_file:
                self.unlink()  # sibling deregisters in the glob->read window
            return real_read(self)

        async with register(vanishing), register(info):
            monkeypatch.setattr(Path, 'read_bytes', racing_read)
            found = find_one(info.session_id, info.name)
        assert found is not None
        assert found.mcp_pid == info.mcp_pid


@pytest.fixture
def info() -> McpServerInfo:
    return McpServerInfo(
        name='test-mcp',
        mcp_pid=os.getpid(),
        claude_pid=999,
        session_id='test-session-abc',
        claude_version=CCVersion('2.1.138'),
        created_at=os_process.create_time(os.getpid()),
        sock_path=None,
        capabilities=['bridge'],
    )


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr('cc_lib.mcp.server_registry.get_claude_workspace_config_home_dir', lambda: tmp_path)
    return tmp_path
