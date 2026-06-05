"""Exception hierarchy for cc_lib.

CCLibError                                # base for all cc_lib exceptions
├── ClaudeContextError                    # session-context failures
│   ├── ClaudeProcessNotFoundError        # no Claude ancestor in process tree
│   │                                     # OR metadata lacks process_created_at anchor
│   ├── SessionNotFoundError              # session not in sessions.json
│   ├── InactiveSessionError              # session in sessions.json but state != 'active'
│   ├── MultipleActiveSessionsForPidError # multiple sessions claim the same claude_pid
│   └── MissingEnvVarError                # expected env var not set
├── HookTreeMismatchError                 # hook tree != CLAUDE_EXEC_LAUNCH_DIR/hooks
└── RivalSessionError                     # another live claude process owns this session_id
"""

from __future__ import annotations

from pathlib import Path

from cc_lib.picklable import PickleByInitArgs

__all__ = [
    'CCLibError',
    'ClaudeContextError',
    'ClaudeProcessNotFoundError',
    'HookTreeMismatchError',
    'InactiveSessionError',
    'MissingEnvVarError',
    'MultipleActiveSessionsForPidError',
    'RivalSessionError',
    'SessionNotFoundError',
]


class CCLibError(Exception):
    """Base for all cc_lib exceptions."""


class ClaudeContextError(CCLibError):
    """Failure resolving Claude session context."""


class ClaudeProcessNotFoundError(ClaudeContextError):
    """Could not materialize a ClaudeProcess.

    Raised by ``find_claude_process`` when the codesign-verified parent walk
    finds no Claude binary, and by ``ClaudeProcess.from_session_metadata`` when
    persisted metadata lacks the ``process_created_at`` anchor.
    """


class SessionNotFoundError(ClaudeContextError):
    """Active session not found in sessions.json."""


class InactiveSessionError(ClaudeContextError):
    """Session exists in sessions.json but is not in 'active' state."""


class MultipleActiveSessionsForPidError(ClaudeContextError):
    """Multiple sessions in sessions.json claim the same active claude_pid.

    Workspace-state invariant violation — should not occur in normal operation.
    """


class MissingEnvVarError(PickleByInitArgs, ClaudeContextError):
    """A required environment variable is not set."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} is not set; this caller expects Bash-tool subprocess context')
        self.var_name = var_name


class HookTreeMismatchError(PickleByInitArgs, CCLibError):
    """Hook script's location does not match CLAUDE_EXEC_LAUNCH_DIR/hooks."""

    def __init__(self, *, actual: Path, expected: Path) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(
            f'hook at {actual} does not match CLAUDE_EXEC_LAUNCH_DIR/hooks ({expected}). '
            f'Hooks are wired to a different tree than CLAUDE_EXEC_LAUNCH_DIR points to.'
        )


class RivalSessionError(PickleByInitArgs, CCLibError):
    """Another live claude process owns this session_id."""

    def __init__(self, *, session_id: str, rival_pid: int, claude_pid: int) -> None:
        self.session_id = session_id
        self.rival_pid = rival_pid
        self.claude_pid = claude_pid
        super().__init__(
            f'session_id {session_id} is owned by live pid {rival_pid} '
            f'(this process is pid {claude_pid}). Refusing to modify the entry.'
        )
