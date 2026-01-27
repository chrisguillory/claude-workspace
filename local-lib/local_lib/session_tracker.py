#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pydantic>=2.0.0",
#   "psutil",
# ]
# ///
"""Session lifecycle tracking utility for Claude Code sessions."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pydantic
from filelock import FileLock

from local_lib.types import JsonDatetime, SessionSource, SessionState

__all__ = [
    'SessionManager',
    'SessionDatabase',
    'Session',
    'SessionMetadata',
]


# Path configuration
SESSIONS_PATH = Path('~/.claude-workspace/sessions.json').expanduser()
LOCK_PATH = Path('~/.claude-workspace/.sessions.json.lock').expanduser()


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)


class SessionDatabase(BaseModel):
    """Container for all tracked sessions."""

    sessions: Sequence[Session] = []


class Session(BaseModel):
    """Claude Code session."""

    # Identity
    session_id: str

    # Current status
    state: SessionState

    # Location
    project_dir: str  # cwd from hook (same as project root)
    transcript_path: str  # Path to session JSONL file

    # Origin
    source: SessionSource

    # Detailed information
    metadata: SessionMetadata


class SessionMetadata(BaseModel):
    """Derived session information."""

    claude_pid: int  # Found via process tree walking
    process_created_at: JsonDatetime | None = None  # When OS created Claude process (from psutil)
    session_ended_at: JsonDatetime | None = None  # When SessionEnd hook fired (clean exit)
    session_end_reason: str | None = None  # Why session ended (prompt_input_exit, clear, logout, other)
    parent_id: str | None = None  # Extracted from transcript file
    crash_detected_at: JsonDatetime | None = None  # When crash detected
    startup_model: str | None = None  # Initial AI model (only set on startup, not resume)
    claude_version: str | None = None  # Claude Code CLI version (from executable symlink path)


class SessionManager:
    """Manages sessions with Unit of Work pattern and file locking.

    Provides atomic session operations with exclusive file locking to prevent
    race conditions. All changes are loaded once on entry and saved once on exit.

    Usage:
        with SessionManager(project_dir) as manager:
            manager.start_session(...)
            manager.detect_crashed_sessions()
    """

    def __init__(self, cwd: str):
        self.cwd = cwd
        self.db_path = get_sessions_file(cwd)
        self.lock_path = LOCK_PATH
        self._db: SessionDatabase | None = None
        self._lock: FileLock | None = None

    def __enter__(self) -> SessionManager:
        """Acquire exclusive lock and load sessions once."""
        self._lock = FileLock(self.lock_path)
        self._lock.acquire()
        self._db = load_sessions(self.cwd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Save on success, rollback on error, always release lock."""
        try:
            if exc_type is None and self._db is not None:
                save_sessions(self.cwd, self._db)
        finally:
            if self._lock:
                self._lock.release()
            self._db = None

    def start_session(
        self,
        session_id: str,
        transcript_path: str,
        source: SessionSource,
        claude_pid: int,
        parent_id: str | None,
        startup_model: str | None = None,
        claude_version: str | None = None,
        process_created_at: datetime | None = None,
    ) -> None:
        """Start a new session or restart an exited/completed/crashed session.

        Args:
            session_id: Session UUID
            transcript_path: Path to session JSONL file
            source: Session source (startup, resume, compact, clear)
            claude_pid: Claude process PID
            parent_id: Parent conversation UUID if available
            startup_model: Initial AI model (only provided on startup, not resume)
            claude_version: Claude Code CLI version (from executable symlink path)
            process_created_at: When OS created Claude process (from psutil)
        """
        if self._db is None:
            raise RuntimeError("SessionManager must be used within 'with' context")

        # Check if session already exists - if so, restart it with new metadata
        for existing_session in self._db.sessions:
            if existing_session.session_id == session_id:
                # Create new session with updated fields (always update when start_session is called)
                restarted_session = Session(
                    session_id=existing_session.session_id,
                    project_dir=existing_session.project_dir,
                    transcript_path=existing_session.transcript_path,
                    source=source,  # Updated to new source (e.g., "resume")
                    state='active',
                    metadata=SessionMetadata(
                        claude_pid=claude_pid,
                        process_created_at=process_created_at,
                        parent_id=parent_id if parent_id is not None else existing_session.metadata.parent_id,
                        startup_model=existing_session.metadata.startup_model,  # Preserve from original
                        claude_version=claude_version
                        if claude_version is not None
                        else existing_session.metadata.claude_version,
                    ),
                )
                # Replace in sessions list
                self._db.sessions = [restarted_session if s.session_id == session_id else s for s in self._db.sessions]
                return

        # Create new session if doesn't exist
        new_session = Session(
            session_id=session_id,
            project_dir=self.cwd,
            transcript_path=transcript_path,
            source=source,
            state='active',
            metadata=SessionMetadata(
                claude_pid=claude_pid,
                process_created_at=process_created_at,
                parent_id=parent_id,
                startup_model=startup_model,
                claude_version=claude_version,
            ),
        )

        # Append to sessions
        self._db.sessions = [*self._db.sessions, new_session]

    def remove_empty_session(self, session_id: str, transcript_path: str) -> bool:
        """Remove a session that has no transcript file.

        Use when session ends without user interaction (no messages sent).

        Args:
            session_id: Session UUID to remove
            transcript_path: Path to transcript file (validated to not exist)

        Returns:
            True if session was found and removed, False if session not found

        Raises:
            ValueError: If transcript file exists (session has content)
        """
        if self._db is None:
            raise RuntimeError("SessionManager must be used within 'with' context")

        if Path(transcript_path).exists():
            raise ValueError(f'Cannot remove session with existing transcript: {transcript_path}')

        original_count = len(self._db.sessions)
        self._db.sessions = [s for s in self._db.sessions if s.session_id != session_id]
        return len(self._db.sessions) < original_count

    def end_session(self, session_id: str, reason: str | None = None) -> None:
        """Mark a session as exited.

        Args:
            session_id: Session UUID to end
            reason: Why session ended (prompt_input_exit, clear, logout, other)
        """
        if self._db is None:
            raise RuntimeError("SessionManager must be used within 'with' context")

        # Find session and create updated version
        for existing_session in self._db.sessions:
            if existing_session.session_id == session_id:
                # Create new session with exited state
                exited_session = Session(
                    session_id=existing_session.session_id,
                    project_dir=existing_session.project_dir,
                    transcript_path=existing_session.transcript_path,
                    source=existing_session.source,
                    state='exited',
                    metadata=SessionMetadata(
                        claude_pid=existing_session.metadata.claude_pid,
                        process_created_at=existing_session.metadata.process_created_at,
                        session_ended_at=datetime.now(UTC).astimezone(),
                        session_end_reason=reason,
                        parent_id=existing_session.metadata.parent_id,
                        crash_detected_at=existing_session.metadata.crash_detected_at,
                        startup_model=existing_session.metadata.startup_model,
                        claude_version=existing_session.metadata.claude_version,
                    ),
                )
                # Replace in sessions list
                self._db.sessions = [exited_session if s.session_id == session_id else s for s in self._db.sessions]
                return

    def detect_crashed_sessions(self) -> Sequence[str]:
        """Check all active sessions and mark crashed if PID is dead.

        Returns:
            Sequence of crashed session IDs
        """
        if self._db is None:
            raise RuntimeError("SessionManager must be used within 'with' context")

        crashed_ids: list[str] = []
        updated_sessions: list[Session] = []

        for session in self._db.sessions:
            if session.state != 'active':
                updated_sessions.append(session)
                continue

            if not psutil.pid_exists(session.metadata.claude_pid):
                # Create new session with crashed state
                crashed_session = Session(
                    session_id=session.session_id,
                    project_dir=session.project_dir,
                    transcript_path=session.transcript_path,
                    source=session.source,
                    state='crashed',
                    metadata=SessionMetadata(
                        claude_pid=session.metadata.claude_pid,
                        process_created_at=session.metadata.process_created_at,
                        session_ended_at=session.metadata.session_ended_at,
                        session_end_reason=session.metadata.session_end_reason,
                        parent_id=session.metadata.parent_id,
                        crash_detected_at=datetime.now(UTC).astimezone(),
                        startup_model=session.metadata.startup_model,
                        claude_version=session.metadata.claude_version,
                    ),
                )
                updated_sessions.append(crashed_session)
                crashed_ids.append(session.session_id)
            else:
                updated_sessions.append(session)

        self._db.sessions = updated_sessions
        return crashed_ids

    def prune_orphaned_sessions(self) -> Sequence[str]:
        """Remove sessions whose transcript files no longer exist.

        Checks all sessions regardless of state and removes any with missing transcripts.

        Returns:
            Sequence of removed session IDs
        """
        if self._db is None:
            raise RuntimeError("SessionManager must be used within 'with' context")

        removed_ids: list[str] = []
        kept_sessions: list[Session] = []

        for session in self._db.sessions:
            if Path(session.transcript_path).exists():
                kept_sessions.append(session)
            else:
                removed_ids.append(session.session_id)

        self._db.sessions = kept_sessions
        return removed_ids


# Deprecated: Old functional API - use SessionManager instead
def add_session(
    cwd: str,
    session_id: str,
    transcript_path: str,
    source: SessionSource,
    claude_pid: int,
    parent_id: str | None,
) -> None:
    """Add a new session to tracking.

    Args:
        cwd: Current working directory path (project root)
        session_id: Session UUID
        transcript_path: Path to session JSONL file
        source: Session source
        claude_pid: Claude process PID
        parent_id: Parent conversation UUID if available
    """
    db = load_sessions(cwd)

    # Check if session already exists
    for session in db.sessions:
        if session.session_id == session_id:
            return

    # Create new session with metadata
    new_session = Session(
        session_id=session_id,
        project_dir=cwd,
        transcript_path=transcript_path,
        source=source,
        state='active',
        metadata=SessionMetadata(
            claude_pid=claude_pid,
            parent_id=parent_id,
        ),
    )

    db.sessions = [*db.sessions, new_session]
    save_sessions(cwd, db)


def end_session(cwd: str, session_id: str) -> None:
    """Mark a session as exited.

    Args:
        cwd: Current working directory path
        session_id: Session UUID to end
    """
    db = load_sessions(cwd)

    # Find and update session
    for session in db.sessions:
        if session.session_id == session_id:
            session.state = 'exited'
            session.metadata.session_ended_at = datetime.now(UTC).astimezone()
            break

    save_sessions(cwd, db)


def get_active_sessions(cwd: str) -> Sequence[Session]:
    """Get all active sessions.

    Args:
        cwd: Current working directory path

    Returns:
        Sequence of active Session objects
    """
    db = load_sessions(cwd)
    return [s for s in db.sessions if s.state == 'active']


def get_exited_sessions(cwd: str) -> Sequence[Session]:
    """Get all exited sessions.

    Args:
        cwd: Current working directory path

    Returns:
        Sequence of exited Session objects
    """
    db = load_sessions(cwd)
    return [s for s in db.sessions if s.state == 'exited']


def get_crashed_sessions(cwd: str) -> Sequence[Session]:
    """Get all crashed sessions.

    Args:
        cwd: Current working directory path

    Returns:
        Sequence of crashed Session objects
    """
    db = load_sessions(cwd)
    return [s for s in db.sessions if s.state == 'crashed']


def detect_crashed_sessions(cwd: str) -> Sequence[str]:
    """Check all active sessions and mark crashed if PID is dead.

    Uses psutil.pid_exists() which is O(1) per check (microseconds to low milliseconds).
    Efficient enough to check hundreds of PIDs without noticeably blocking execution.

    Args:
        cwd: Current working directory path

    Returns:
        Sequence of crashed session IDs
    """
    db = load_sessions(cwd)
    crashed_ids: list[str] = []

    for session in db.sessions:
        if session.state != 'active':
            continue

        # psutil.pid_exists() is fast - O(1) syscall, not scanning all processes
        if not psutil.pid_exists(session.metadata.claude_pid):
            session.state = 'crashed'
            session.metadata.crash_detected_at = datetime.now(UTC).astimezone()
            crashed_ids.append(session.session_id)

    if crashed_ids:
        save_sessions(cwd, db)

    return crashed_ids


# Private helper functions (not in __all__)
def get_sessions_file(cwd: str) -> Path:
    """Get path to sessions.json file - centralized in ~/.claude-workspace/."""
    SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    return SESSIONS_PATH


def load_sessions(cwd: str) -> SessionDatabase:
    """Load sessions from sessions.json file.

    Args:
        cwd: Current working directory path

    Returns:
        SessionDatabase with all tracked sessions
    """
    sessions_file = get_sessions_file(cwd)

    if not sessions_file.exists():
        return SessionDatabase(sessions=[])

    with open(sessions_file) as f:
        data = json.load(f)
        return SessionDatabase.model_validate(data)


def save_sessions(cwd: str, db: SessionDatabase) -> None:
    """Save sessions to sessions.json file using atomic write.

    Uses temp file + rename for atomic operation to prevent corruption.

    Args:
        cwd: Current working directory path
        db: SessionDatabase to save
    """
    sessions_file = get_sessions_file(cwd)

    # Write to temp file in same directory (required for atomic rename)
    with tempfile.NamedTemporaryFile(mode='w', dir=sessions_file.parent, delete=False, suffix='.tmp') as f:
        temp_path = Path(f.name)
        json.dump(db.model_dump(mode='json', exclude_none=True), f, indent=2)

    # Atomic rename
    temp_path.replace(sessions_file)
