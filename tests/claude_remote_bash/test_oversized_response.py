"""End-to-end regression: oversize ExecuteResult becomes a legible ErrorResponse, not IncompleteReadError.

When a command's stdout pushes the JSON-serialized ExecuteResult above the
10 MB wire-frame limit (protocol.py: MAX_PAYLOAD_SIZE), the daemon must
substitute an ErrorResponse on the same connection rather than aborting
the connection mid-frame.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_remote_bash import auth as auth_module
from claude_remote_bash import paths as paths_module
from claude_remote_bash.auth import DaemonConfig
from claude_remote_bash.daemon import _Daemon
from claude_remote_bash.protocol import MAX_PAYLOAD_SIZE, read_message, write_message
from claude_remote_bash.schemas.protocol import (
    AuthOk,
    AuthRequest,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
)


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
async def running_daemon(isolated_config: Path) -> AsyncIterator[int]:
    _ = isolated_config
    config = DaemonConfig(
        name='test-daemon',
        auth_key='deadbeef' * 8,
        shell='/bin/zsh',
        session_timeout_minutes=1440,
    )
    daemon = _Daemon(config)
    server = await asyncio.start_server(daemon.handle_client, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


# -- Tests -------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversize_execute_result_becomes_error_response(
    running_daemon: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ExecuteResult whose JSON exceeds MAX_PAYLOAD_SIZE returns ErrorResponse, not IncompleteReadError."""
    port = running_daemon
    oversize_stdout = 'A' * (MAX_PAYLOAD_SIZE + 1024 * 1024)  # ~11 MB

    async def fake_handle_execute(self: _Daemon, msg: ExecuteRequest) -> ExecuteResult:
        return ExecuteResult(
            id=msg.id,
            stdout=oversize_stdout,
            stderr='',
            exit_code=0,
            cwd='/tmp',
        )

    monkeypatch.setattr(_Daemon, '_handle_execute', fake_handle_execute)

    reader, writer = await _authed_conn(port)
    try:
        await write_message(
            writer,
            ExecuteRequest(id='req-oversize', command='echo big', session_id='s1', agent_id=None, timeout=None),
        )
        resp = await asyncio.wait_for(read_message(reader), timeout=5.0)

        assert isinstance(resp, ErrorResponse), f'expected ErrorResponse, got {type(resp).__name__}'
        assert resp.id == 'req-oversize'
        assert 'response too large for wire frame' in resp.message
        assert 'payload too large' in resp.message
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_normal_execute_after_oversize_works_on_same_connection(
    running_daemon: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the oversize-refusal, the same connection remains usable for further requests."""
    port = running_daemon
    call_count = {'n': 0}

    async def fake_handle_execute(self: _Daemon, msg: ExecuteRequest) -> ExecuteResult:
        call_count['n'] += 1
        if call_count['n'] == 1:
            return ExecuteResult(
                id=msg.id,
                stdout='A' * (MAX_PAYLOAD_SIZE + 1024 * 1024),
                stderr='',
                exit_code=0,
                cwd='/tmp',
            )
        return ExecuteResult(id=msg.id, stdout='small', stderr='', exit_code=0, cwd='/tmp')

    monkeypatch.setattr(_Daemon, '_handle_execute', fake_handle_execute)

    reader, writer = await _authed_conn(port)
    try:
        await write_message(
            writer,
            ExecuteRequest(id='req-1', command='echo big', session_id='s1', agent_id=None, timeout=None),
        )
        resp1 = await asyncio.wait_for(read_message(reader), timeout=5.0)
        assert isinstance(resp1, ErrorResponse)
        assert resp1.id == 'req-1'

        await write_message(
            writer,
            ExecuteRequest(id='req-2', command='echo small', session_id='s1', agent_id=None, timeout=None),
        )
        resp2 = await asyncio.wait_for(read_message(reader), timeout=3.0)
        assert isinstance(resp2, ExecuteResult)
        assert resp2.id == 'req-2'
        assert resp2.stdout == 'small'
    finally:
        writer.close()
        await writer.wait_closed()


# -- Private helpers ---------------------------------------------------


async def _authed_conn(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    await write_message(writer, AuthRequest(key='deadbeef' * 8))
    resp = await asyncio.wait_for(read_message(reader), timeout=3.0)
    assert isinstance(resp, AuthOk), f'expected AuthOk, got {resp}'
    return reader, writer
