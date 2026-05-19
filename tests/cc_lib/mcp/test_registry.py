"""Tests for cc_lib.mcp.registry -- McpServerInfo, register (CM), read_all, clear_session."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cc_lib.mcp.registry import McpServerInfo, clear_session, read_all, register


class TestRegistry:
    """Registry edges + regression-bait that empirical adoption won't catch."""

    async def test_empty_session(self) -> None:
        assert list(read_all('no-such-session')) == []

    async def test_round_trip_preserves_info(self, info: McpServerInfo) -> None:
        """JSON serialize -> deserialize -> equality. Schema-change regression bait."""
        async with register(info):
            assert list(read_all(info.session_id)) == [info]

    async def test_skips_dead_pid(self, info: McpServerInfo, monkeypatch: pytest.MonkeyPatch) -> None:
        """The dispositive read_all behavior — without this, stale entries leak."""
        async with register(info):
            monkeypatch.setattr('cc_lib.mcp.registry.psutil.pid_exists', lambda _pid: False)
            assert list(read_all(info.session_id)) == []

    async def test_skips_corrupt_file(self, info: McpServerInfo, workspace_dir: Path) -> None:
        registry_dir = workspace_dir / 'mcp' / 'registry' / info.session_id
        registry_dir.mkdir(parents=True)
        (registry_dir / 'corrupt-1.json').write_text('not valid json')
        async with register(info):
            assert list(read_all(info.session_id)) == [info]

    async def test_register_cleans_up_on_exit(self, info: McpServerInfo, workspace_dir: Path) -> None:
        """File written on enter, removed on exit. The CM contract."""
        path = workspace_dir / 'mcp' / 'registry' / info.session_id / f'{info.name}-{info.mcp_pid}.json'
        async with register(info):
            assert path.is_file()
        assert not path.exists()

    def test_clear_missing_session_is_no_op(self) -> None:
        clear_session('no-such-session')  # must not raise


@pytest.fixture
def info() -> McpServerInfo:
    return McpServerInfo(
        name='test-mcp',
        mcp_pid=os.getpid(),
        claude_pid=999,
        session_id='test-session-abc',
        claude_version='2.1.138',
        started_at=datetime.now(UTC),
    )


@pytest.fixture(autouse=True)
def workspace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr('cc_lib.mcp.registry.get_claude_workspace_config_home_dir', lambda: tmp_path)
    return tmp_path
