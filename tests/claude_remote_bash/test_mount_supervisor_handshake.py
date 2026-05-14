"""Empirical tests for the parent-side IPC reader used by ``spawn_detached_supervisor``."""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from typing import IO

import pytest
from claude_remote_bash.mount import _read_first_line_with_timeout


def test_read_first_line_returns_the_line_immediately() -> None:
    stream: IO[str] = io.StringIO('READY mid-abcdef\n')
    line = _read_first_line_with_timeout(stream, timeout=2.0)
    assert line == 'READY mid-abcdef\n'


def test_read_first_line_returns_none_on_eof() -> None:
    """Supervisor exited before writing — readline returns '' → we return None."""
    stream: IO[str] = io.StringIO('')
    line = _read_first_line_with_timeout(stream, timeout=2.0)
    assert line is None


def test_read_first_line_raises_timeout_when_stream_blocks() -> None:
    """A stream that never produces a line should hit the timeout."""
    stream = _BlockingStream()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        _read_first_line_with_timeout(stream, timeout=0.3)
    elapsed = time.monotonic() - start
    assert 0.2 <= elapsed < 1.0, f'timeout fired at {elapsed:.2f}s, expected ~0.3s'


# -- Test helpers --------------------------------------------------------------


class _BlockingStream:
    """A stream whose ``readline`` blocks forever — exercises the timeout path."""

    closed = False

    def readline(self, limit: int = -1) -> str:
        _ = limit  # unused; matches IO[str].readline signature
        _block_forever()
        return ''

    def readlines(self, hint: int = -1) -> Sequence[str]:
        _ = hint
        return []


def _block_forever() -> None:
    while True:
        time.sleep(0.1)
