"""Pure OS-process primitives, encapsulating psutil; consumers don't import psutil for these."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import psutil

# Re-exports so consumers catch by os_process.* names without importing psutil.
ProcessAccessDenied = psutil.AccessDenied
ProcessGone = psutil.NoSuchProcess
ProcessZombie = psutil.ZombieProcess

__all__ = [
    'ProcessAccessDenied',
    'ProcessGone',
    'ProcessHandle',
    'ProcessZombie',
    'create_time',
    'exe_path',
    'is_alive',
    'terminate',
    'try_exe_path',
]


class ProcessHandle(NamedTuple):
    """A (pid, created_at) anchor for recycle-safe liveness checks.

    The created_at value pins process identity at capture; ``is_alive()``
    refuses PIDs whose create_time has drifted (recycled by a different process).
    """

    pid: int
    created_at: datetime

    @classmethod
    def capture(cls, pid: int) -> ProcessHandle | None:
        """Snapshot ``(pid, create_time)``. Returns None if the PID is already dead."""
        try:
            return cls(pid, create_time(pid))
        except ProcessGone:
            return None

    def is_alive(self, *, tolerance: float = 0.0) -> bool:
        """True iff the process is alive AND create_time matches the anchor within tolerance.

        Default ``tolerance=0.0`` (byte-equality) is empirically stable on
        macOS arm64 + Linux psutil 7.x — successive ``create_time()`` calls
        and ISO datetime roundtrips produce identical floats. Pass a small
        positive value only when reading snapshots from a different psutil
        version or OS.
        """
        if not is_alive(self.pid):
            return False
        try:
            actual = create_time(self.pid)
        except ProcessGone:
            return False
        return abs((actual - self.created_at).total_seconds()) <= tolerance


def is_alive(pid: int) -> bool:
    """True if process with this PID exists (any state, including zombie)."""
    return psutil.pid_exists(pid)


def create_time(pid: int) -> datetime:
    """UTC datetime of process creation. Raises ProcessGone if dead."""
    return datetime.fromtimestamp(psutil.Process(pid).create_time(), tz=UTC)


def exe_path(pid: int) -> Path:
    """Process executable path. Raises ProcessGone if dead, ProcessAccessDenied for kernel pids (0)."""
    return Path(psutil.Process(pid).exe())


def try_exe_path(pid: int) -> Path | None:
    """Probe variant of ``exe_path`` — ``None`` if the pid is gone or protected."""
    try:
        return exe_path(pid)
    except (ProcessGone, ProcessAccessDenied):
        return None


def terminate(pid: int) -> None:
    """SIGTERM the process. Raises ProcessGone if already dead, ProcessAccessDenied for protected pids (1)."""
    psutil.Process(pid).terminate()
