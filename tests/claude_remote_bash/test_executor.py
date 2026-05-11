"""Tests for `claude_remote_bash.executor.execute_command`.

These tests exercise the full shell wrapping + marker parsing pipeline
against a real ``/bin/zsh`` subprocess. They do not require the daemon,
mDNS, or auth — just a zsh binary on PATH.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from claude_remote_bash.executor import CommandResult, execute_command


class TestExecuteCommand:
    """End-to-end tests covering the shell wrap + marker round-trip."""

    def test_normal_stdout_with_newline(self) -> None:
        result = _run(execute_command('echo hi'))
        assert result.stdout == 'hi'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_empty_stdout(self) -> None:
        result = _run(execute_command('true'))
        assert result.stdout == ''
        assert result.exit_code == 0

    def test_nonzero_exit_code(self) -> None:
        result = _run(execute_command('false'))
        assert result.exit_code == 1

    def test_stdout_without_trailing_newline_regression(self, tmp_path: Path) -> None:
        """A command whose stdout lacks a trailing newline must not glue the marker on.

        Regression for the bug where ``cat`` of a file without a final ``\\n``
        produced output like ``main()__CRBD_<hex>_0_/Users/...`` because the
        trap's ``echo`` was concatenated to the file's last line.
        """
        f = tmp_path / 'no-newline.txt'
        f.write_bytes(b'main()')  # exactly 6 bytes, no \n
        result = _run(execute_command(f'cat {f}'))
        assert result.stdout == 'main()'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_multiline_stdout_no_trailing_newline(self) -> None:
        result = _run(execute_command('printf "a\\nb"'))
        assert result.stdout == 'a\nb'
        assert '__CRBD_' not in result.stdout
        assert result.exit_code == 0

    def test_stderr_passes_through(self) -> None:
        result = _run(execute_command('echo err 1>&2'))
        assert result.stdout == ''
        assert result.stderr == 'err'
        assert result.exit_code == 0

    def test_cwd_tracking_after_cd(self) -> None:
        result = _run(execute_command('cd /tmp && pwd'))
        # macOS resolves /tmp to /private/tmp; either is acceptable.
        assert result.cwd in {'/tmp', '/private/tmp'}

    def test_exit_trap_fires_on_explicit_exit(self) -> None:
        """Even when the command calls ``exit``, the trap still emits the marker."""
        result = _run(execute_command('echo before; exit 7'))
        assert result.stdout == 'before'
        assert result.exit_code == 7


def _run(coro: Coroutine[Any, Any, CommandResult]) -> CommandResult:
    return asyncio.new_event_loop().run_until_complete(coro)
