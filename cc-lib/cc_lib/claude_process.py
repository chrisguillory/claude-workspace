"""ClaudeProcess class — codesign-verified Claude Code OS process abstraction."""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cc_lib import os_process
from cc_lib.exceptions import ClaudeProcessNotFoundError
from cc_lib.os_process import ProcessHandle
from cc_lib.session_tracker import SessionMetadata
from cc_lib.settings_env import claude_binary_name

if TYPE_CHECKING:
    from cc_lib.claude_context import ClaudeContext

__all__ = [
    'ClaudeProcess',
    'kill_and_copy_resume',
]


class ClaudeProcess:
    """A Claude Code OS process with codesign-verified provenance + recycle anchor.

    Construction implies codesign verification has been performed (either at
    ``from_pid`` time, or at write-time for ``from_session_metadata``). The
    instance carries ``(pid, created_at)`` — ``created_at`` is the recycle
    anchor; ``is_alive`` refuses the PID if its current create_time has drifted
    (PID was recycled by a different process).
    """

    __slots__ = ('_handle',)

    @classmethod
    def from_pid(cls, pid: int) -> ClaudeProcess | None:
        """Codesign-verify ``pid`` and capture its create_time anchor.

        Returns:
            ClaudeProcess — pid is codesign-verified Claude AND alive
            None          — not Claude OR PID is gone OR codesign unavailable

        Used by ``find_claude_process`` in the parent-walk iteration; the None
        return is the "continue walking" signal.
        """
        if not _is_claude_pid(pid):
            return None
        handle = ProcessHandle.capture(pid)
        if handle is None:
            return None
        return cls(handle)

    @classmethod
    def from_session_metadata(cls, metadata: SessionMetadata) -> ClaudeProcess:
        """Reconstruct from persisted ``(claude_pid, process_created_at)``.

        Does NOT re-codesign-verify — write-time was the gate when
        SessionStart hook captured the metadata via ``find_claude_process``.

        Raises:
            ClaudeProcessNotFoundError — metadata lacks the ``process_created_at`` anchor.
        """
        if metadata.process_created_at is None:
            raise ClaudeProcessNotFoundError(
                f'Session metadata for pid {metadata.claude_pid} lacks process_created_at anchor'
            )
        return cls(ProcessHandle(metadata.claude_pid, metadata.process_created_at))

    def __init__(self, handle: ProcessHandle) -> None:
        self._handle = handle

    @property
    def pid(self) -> int:
        return self._handle.pid

    @property
    def created_at(self) -> datetime:
        return self._handle.created_at

    def exe_path(self) -> Path:
        """Query the running process's executable path. Raises ``os_process.ProcessGone`` if dead.

        Not a property — costs a syscall and can raise; the parens at call sites
        signal that work happens.
        """
        return os_process.exe_path(self.pid)

    def is_alive(self, *, tolerance: float = 0.0) -> bool:
        """True iff alive AND create_time matches anchor within ``tolerance``.

        Default ``tolerance=0.0`` (byte-equality) — recycle defense.
        """
        return self._handle.is_alive(tolerance=tolerance)

    def terminate(self) -> None:
        """SIGTERM the process. No-op if PID was recycled or already dead."""
        if self.is_alive():
            os_process.terminate(self.pid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClaudeProcess) and other._handle == self._handle

    def __hash__(self) -> int:
        return hash(self._handle)

    def __repr__(self) -> str:
        return f'ClaudeProcess(pid={self.pid}, created_at={self.created_at.isoformat()})'


def kill_and_copy_resume(
    claude_context: ClaudeContext,
    *,
    extra_args: Sequence[str] = (),
) -> str:
    """Kill Claude Code and copy ``{claude_binary_name} --resume {session-id}`` to clipboard.

    SIGTERM fires 0.5s after this returns (detached subprocess), giving the
    caller time to print success messages before Claude exits.
    """
    parts = [
        claude_binary_name(),
        '--resume',
        shlex.quote(claude_context.session_id),
        *(shlex.quote(a) for a in extra_args),
    ]
    resume_cmd = ' '.join(parts)

    subprocess.run(['pbcopy'], input=resume_cmd.encode(), check=False)

    subprocess.Popen(
        [sys.executable, '-c', _KILL_SCRIPT, str(claude_context.claude_pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    return resume_cmd


def _is_claude_pid(pid: int) -> bool:
    """True if ``pid`` runs the Claude Code binary (codesign Identifier verified).

    Anthropic embeds ``Identifier=com.anthropic.claude-code`` in the binary's
    code signature, bound to the code-directory hash. The Identifier is
    preserved across the patcher's adhoc re-sign. macOS-only.
    """
    result = subprocess.run(
        ['codesign', '-dvv', f'+{pid}'],
        check=False,
        capture_output=True,
        text=True,
    )
    # codesign writes the Identifier= line to stderr.
    return 'Identifier=com.anthropic.claude-code' in result.stderr.splitlines()


_KILL_SCRIPT = """\
import os, signal, sys, time
pid = int(sys.argv[1])
time.sleep(0.5)
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except OSError:
        break
"""
