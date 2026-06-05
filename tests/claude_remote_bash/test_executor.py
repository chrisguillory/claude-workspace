"""Tests for `claude_remote_bash.executor.execute_command`.

These tests exercise the full shell wrapping + marker parsing pipeline
against a real ``/bin/zsh`` subprocess. They do not require the daemon,
mDNS, or auth — just a zsh binary on PATH.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from claude_remote_bash.executor import execute_command


class TestExecuteCommand:
    """End-to-end tests covering the shell wrap + marker round-trip."""

    def test_normal_stdout_with_newline(self) -> None:
        result = asyncio.run(execute_command('echo hi'))
        assert result.stdout == 'hi'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_empty_stdout(self) -> None:
        result = asyncio.run(execute_command('true'))
        assert result.stdout == ''
        assert result.exit_code == 0

    def test_nonzero_exit_code(self) -> None:
        result = asyncio.run(execute_command('false'))
        assert result.exit_code == 1

    def test_stdout_without_trailing_newline_regression(self, tmp_path: Path) -> None:
        """A command whose stdout lacks a trailing newline must not glue the marker on.

        Regression for the bug where ``cat`` of a file without a final ``\\n``
        produced output like ``main()__CRBD_<hex>_0_/Users/...`` because the
        trap's ``echo`` was concatenated to the file's last line.
        """
        f = tmp_path / 'no-newline.txt'
        f.write_bytes(b'main()')  # exactly 6 bytes, no \n
        result = asyncio.run(execute_command(f'cat {f}'))
        assert result.stdout == 'main()'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_multiline_stdout_no_trailing_newline(self) -> None:
        result = asyncio.run(execute_command('printf "a\\nb"'))
        assert result.stdout == 'a\nb'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_stderr_passes_through(self) -> None:
        result = asyncio.run(execute_command('echo err 1>&2'))
        assert result.stdout == ''
        assert result.stderr == 'err'
        assert result.exit_code == 0

    def test_cwd_tracking_after_cd(self) -> None:
        result = asyncio.run(execute_command('cd /tmp && pwd'))
        # macOS resolves /tmp to /private/tmp; either is acceptable.
        assert result.cwd in {'/tmp', '/private/tmp'}

    def test_exit_trap_fires_on_explicit_exit(self) -> None:
        """Even when the command calls ``exit``, the trap still emits the marker."""
        result = asyncio.run(execute_command('echo before; exit 7'))
        assert result.stdout == 'before'
        assert result.exit_code == 7

    def test_explicit_timeout_kills_overrun(self) -> None:
        """An explicit numeric timeout still SIGKILLs commands that exceed it."""
        result = asyncio.run(execute_command('sleep 5', timeout=0.5))
        assert result.exit_code == -1
        assert result.stdout == '[TIMEOUT]'

    def test_none_timeout_runs_without_limit(self) -> None:
        """``timeout=None`` skips the ``asyncio.wait_for`` wrapper."""
        result = asyncio.run(execute_command('sleep 1 && echo done', timeout=None))
        assert result.stdout == 'done'
        assert result.exit_code == 0

    def test_default_timeout_is_none(self) -> None:
        """Omitting ``timeout`` runs without limit."""
        result = asyncio.run(execute_command('sleep 1 && echo done'))
        assert result.stdout == 'done'
        assert result.exit_code == 0

    def test_cancellation_kills_subprocess_group(self) -> None:
        """A cancelled ``execute_command`` SIGKILLs the whole subprocess group.

        Without the group kill, daemon shutdown would unblock but the user's
        command and its children would survive as orphans reparented to launchd.
        """
        sentinel = f'crbtest-{uuid.uuid4().hex[:8]}'

        async def _run() -> str:
            task = asyncio.create_task(
                execute_command(f'sleep 47 # {sentinel}', timeout=60),
            )
            pgid = ''
            for _ in range(30):
                await asyncio.sleep(0.1)
                found = subprocess.run(
                    ['pgrep', '-f', sentinel],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if found.stdout.strip():
                    pid = found.stdout.strip().split('\n')[0]
                    pgid_out = subprocess.run(
                        ['ps', '-p', pid, '-o', 'pgid='],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    pgid = pgid_out.stdout.strip()
                    if pgid:
                        break
            if not pgid:
                pytest.fail("subprocess didn't appear within 3s")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return pgid

        pgid = asyncio.run(_run())
        time.sleep(0.5)
        remaining = subprocess.run(
            ['pgrep', '-g', pgid],
            capture_output=True,
            text=True,
            check=False,
        )
        assert remaining.returncode == 1, f'pgrep -g {pgid} returned matches after cancellation: {remaining.stdout!r}'
        assert not remaining.stdout.strip(), f'processes in pgid {pgid} survived cancellation: {remaining.stdout!r}'
