"""
Shared exceptions for claude-session.

Domain-specific exceptions used across services.

Exception Hierarchy:
    ClaudeSessionError (base)
    ├── SessionResolutionError (lookup/resolution failures)
    │   ├── AmbiguousSessionError (prefix matches multiple sessions)
    │   └── SourceProjectConflictError (--source-project with auto-detection)
    ├── SessionDeletionError (deletion policy violations)
    │   ├── NativeSessionDeletionError (native session without --force)
    │   ├── RunningSessionDeletionError (running session without --terminate)
    │   ├── CrossSessionArtifactsRequiredError (sibling copies exist, flag missing)
    │   └── CrossSessionArtifactsNotApplicableError (no siblings, flag passed)
    └── SessionMoveError (move policy violations)
        ├── SameProjectMoveError (session already in target project)
        ├── NativeSessionMoveError (native session without --force)
        └── RunningSessionMoveError (running session without --terminate)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = [
    'AmbiguousSessionError',
    'ClaudeSessionError',
    'CrossSessionArtifactsNotApplicableError',
    'CrossSessionArtifactsRequiredError',
    'NativeSessionDeletionError',
    'NativeSessionMoveError',
    'RunningSessionDeletionError',
    'RunningSessionMoveError',
    'SameProjectMoveError',
    'SessionDeletionError',
    'SessionMoveError',
    'SessionResolutionError',
    'SourceProjectConflictError',
]


class ClaudeSessionError(Exception):
    """Base exception for all claude-session errors."""


class SessionResolutionError(ClaudeSessionError):
    """Base exception for session lookup and resolution failures."""


class AmbiguousSessionError(SessionResolutionError):
    """Raised when a session ID prefix matches multiple sessions."""

    def __init__(self, prefix: str, matches: Sequence[str]) -> None:
        self.prefix = prefix
        self.matches = matches
        matches_str = '\n  '.join(matches[:10])
        if len(matches) > 10:
            matches_str += f'\n  ... and {len(matches) - 10} more'
        super().__init__(
            f"Session ID prefix '{prefix}' is ambiguous. Matches {len(matches)} sessions:\n  {matches_str}\n\n"
            f'Provide a more specific session ID prefix, or use --source-project to target a specific project.'
        )


class SourceProjectConflictError(SessionResolutionError):
    """Raised when --source-project is used with an auto-detected session ID."""

    def __init__(self) -> None:
        super().__init__('--source-project requires an explicit session ID (cannot be used with auto-detection).')


class SessionDeletionError(ClaudeSessionError):
    """Base exception for deletion policy violations."""


class NativeSessionDeletionError(SessionDeletionError):
    """Raised when attempting to delete a native session without --force."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} is a native Claude session (UUIDv4). Use --force to delete native sessions.'
        )


class RunningSessionDeletionError(SessionDeletionError):
    """Raised when attempting to delete a running session without --terminate."""

    def __init__(self, session_id: str, pid: int) -> None:
        self.session_id = session_id
        self.pid = pid
        super().__init__(f'Session {session_id} is running (PID {pid}). Use --terminate to kill the process.')


class CrossSessionArtifactsRequiredError(SessionDeletionError):
    """Raised when siblings exist but delete_cross_session_artifacts is unset."""

    def __init__(self, session_id: str, sibling_project_folders: Sequence[Path]) -> None:
        self.session_id = session_id
        self.sibling_project_folders = sibling_project_folders
        siblings_str = ', '.join(str(p) for p in sibling_project_folders)
        super().__init__(
            f'Session {session_id} has sibling copies in other projects ({siblings_str}). '
            f'Set delete_cross_session_artifacts to true to delete shared artifacts, or false to preserve them.'
        )


class CrossSessionArtifactsNotApplicableError(SessionDeletionError):
    """Raised when delete_cross_session_artifacts is passed but no siblings exist."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} has no sibling copies; delete_cross_session_artifacts has no effect. Omit the argument.'
        )


class SessionMoveError(ClaudeSessionError):
    """Base exception for move policy violations."""


class SameProjectMoveError(SessionMoveError):
    """Raised when attempting to move a session to its current project."""

    def __init__(self, session_id: str, project_path: str) -> None:
        self.session_id = session_id
        self.project_path = project_path
        super().__init__(f'Session {session_id} is already in project {project_path}.')


class NativeSessionMoveError(SessionMoveError):
    """Raised when attempting to move a native session without --force."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f'Session {session_id} is a native Claude session (UUIDv4). Use --force to move native sessions.'
        )


class RunningSessionMoveError(SessionMoveError):
    """Raised when attempting to move a running session without --terminate."""

    def __init__(self, session_id: str, pid: int) -> None:
        self.session_id = session_id
        self.pid = pid
        super().__init__(f'Session {session_id} is running (PID {pid}). Use --terminate to kill the process.')
