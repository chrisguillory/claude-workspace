"""Regression: ProcessHandle.is_alive() rejects a mismatched create_time anchor."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from cc_lib.os_process import ProcessHandle


def test_process_handle_rejects_recycled_pid() -> None:
    """Byte-equality default (tolerance=0.0) must reject anchor drift.

    Locks in the recycle-defense contract: if a future refactor widens the
    default tolerance, the 5 recycle-defense callsites in the workspace
    would silently start accepting recycled PIDs. This test catches that.
    """
    h = ProcessHandle(os.getpid(), datetime.now(UTC) - timedelta(hours=1))
    assert h.is_alive() is False
