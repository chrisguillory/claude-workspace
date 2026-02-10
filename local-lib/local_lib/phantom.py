"""Phantom session detection for Claude Code's resume double-fire bug.

Claude Code >=2.1.37 sends two SessionStart hook events when resuming a session:

  1. A real event:    source='resume', original session ID, transcript file exists
  2. A phantom event: source='startup', NEW session ID, transcript file never created

The phantom's session ID is never used by Claude Code - no JSONL file is written for it.
If left unhandled, it creates an orphaned "active" entry in sessions.json.

ORDERING IS NON-DETERMINISTIC
------------------------------
Both events dispatch concurrently (~14ms apart) and race for the SessionManager file lock.
We observe both orderings in practice:

  Case 1 - Resume wins lock first:
    Resume registers the real session (PID now active).
    Phantom arrives, sees PID already active → PRE-GUARD skips it.

  Case 2 - Phantom wins lock first:
    Phantom registers (no active session with this PID yet → guard misses).
    Resume arrives, start_session() updates the existing real session.
    Now two active entries share the same PID → POST-CLEANUP removes the phantom.

VERSION TRACKING
-----------------
Every phantom detection (either path) is logged to ~/.claude-workspace/phantom_log.json
keyed by Claude Code version. When a future version stops producing phantoms, the log
stops growing - signaling the workaround is dead code and can be removed.

  {
    "2.1.37": {"count": 5, "last_seen": "2026-02-07T13:29:04-08:00"},
    "2.1.38": {"count": 0, "last_seen": null}   <-- bug fixed in this version
  }
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Set
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    from local_lib.session_tracker import SessionManager

PHANTOM_LOG_PATH = Path('~/.claude-workspace/phantom_log.json').expanduser()
PHANTOM_LOCK_PATH = Path('~/.claude-workspace/.phantom_log.json.lock').expanduser()


class PhantomHandler:
    """Detects and removes phantom sessions from Claude Code's resume bug.

    Create inside a SessionManager context. Call is_phantom() before start_session(),
    and cleanup() after. Call log() and print_diagnostics() after the context exits.

    Example:
        with SessionManager(project_dir) as manager:
            manager.detect_crashed_sessions()
            manager.prune_orphaned_sessions()

            phantom = PhantomHandler(manager, claude_pid, claude_version)

            if phantom.is_phantom(session_id, source, transcript_path):
                pass  # Skip registration
            else:
                manager.start_session(...)
                phantom.cleanup(session_id)

        phantom.log()
        phantom.print_diagnostics()
    """

    def __init__(self, manager: SessionManager, claude_pid: int, claude_version: str) -> None:
        self._manager = manager
        self._pid = claude_pid
        self._version = claude_version
        self._skipped: str | None = None
        self._cleaned: Set[str] = set()

    @property
    def detected(self) -> bool:
        """Whether any phantom was detected (either skipped or cleaned)."""
        return self._skipped is not None or bool(self._cleaned)

    def is_phantom(self, session_id: str, source: str, transcript_path: Path) -> bool:
        """PRE-GUARD: Check if this event is a phantom that should be skipped.

        Returns True when all three conditions hold:
          - source is 'startup' (phantoms always claim to be startups)
          - transcript file doesn't exist (phantom's file is never created)
          - another active session already owns this PID (the real resume event
            won the lock and registered first)

        This handles Case 1 (resume fires first). Case 2 is handled by cleanup().
        """
        if (
            source == 'startup'
            and not transcript_path.exists()
            and self._manager.get_active_sessions_for_pid(self._pid)
        ):
            self._skipped = session_id
            return True
        return False

    def cleanup(self, real_session_id: str) -> Set[str]:
        """POST-CLEANUP: Remove phantoms that beat us to the lock.

        Scans active sessions that share our PID but have a different session ID
        and no transcript file. These are phantoms that won the lock race (Case 2).

        Returns:
            Sequence of removed phantom session IDs.
        """
        phantom_ids = {
            s.session_id
            for s in self._manager.get_active_sessions_for_pid(self._pid)
            if s.session_id != real_session_id and not Path(s.transcript_path).exists()
        }
        if phantom_ids:
            self._cleaned = self._manager.remove_sessions(phantom_ids)
        return self._cleaned

    def log(self) -> None:
        """Write phantom detections to the version-keyed tracker.

        Safe to call outside the SessionManager context - uses its own file lock.
        No-op if no phantoms were detected.
        """
        if not self.detected:
            return

        count = (1 if self._skipped else 0) + len(self._cleaned)

        with FileLock(PHANTOM_LOCK_PATH):
            phantom_log: dict[str, dict[str, int | str]] = {}
            if PHANTOM_LOG_PATH.exists():
                phantom_log = json.loads(PHANTOM_LOG_PATH.read_text())

            entry = phantom_log.get(self._version, {'count': 0})
            entry['count'] = int(entry['count']) + count
            entry['last_seen'] = datetime.now(UTC).astimezone().isoformat()
            phantom_log[self._version] = entry

            with tempfile.NamedTemporaryFile(mode='w', dir=PHANTOM_LOG_PATH.parent, delete=False, suffix='.tmp') as f:
                json.dump(phantom_log, f, indent=2)
                temp_path = Path(f.name)
            temp_path.replace(PHANTOM_LOG_PATH)

    def print_diagnostics(self) -> None:
        """Print diagnostic output for hook logging.

        Reports which path caught the phantom:
          - phantom_skipped: PRE-GUARD caught it (resume won lock first)
          - phantom_cleaned: POST-CLEANUP caught it (phantom won lock first)
        """
        if self._skipped:
            print(f'phantom_skipped: {self._skipped} (PID {self._pid} already active, v{self._version})')
        if self._cleaned:
            print(f'phantom_cleaned: {self._cleaned} (same PID {self._pid}, no transcript, v{self._version})')
