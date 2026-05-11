"""Exception hierarchy for cc_lib.

CCLibError                              # base for all cc_lib exceptions
├── ClaudeContextError                  # session-context failures
│   ├── ClaudeProcessNotFoundError      # no Claude ancestor in process tree
│   ├── SessionNotFoundError            # session not in sessions.json
│   ├── AmbiguousSessionError           # multiple matches in sessions.json
│   └── MissingEnvVarError              # expected env var not set
└── HookTreeMismatchError               # hook tree != CLAUDE_EXEC_LAUNCH_DIR/hooks
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    'AmbiguousSessionError',
    'CCLibError',
    'ClaudeContextError',
    'ClaudeProcessNotFoundError',
    'HookTreeMismatchError',
    'MissingEnvVarError',
    'SessionNotFoundError',
]


class CCLibError(Exception):
    """Base for all cc_lib exceptions."""


class ClaudeContextError(CCLibError):
    """Failure resolving Claude session context."""


class ClaudeProcessNotFoundError(ClaudeContextError):
    """No Claude binary found in parent process tree (codesign-verified walk)."""


class SessionNotFoundError(ClaudeContextError):
    """Active session not found in sessions.json."""


class AmbiguousSessionError(ClaudeContextError):
    """Multiple active sessions matched the same query."""


class MissingEnvVarError(ClaudeContextError):
    """A required environment variable is not set."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} is not set; this caller expects Bash-tool subprocess context')
        self.var_name = var_name


class HookTreeMismatchError(CCLibError):
    """Hook script's location does not match CLAUDE_EXEC_LAUNCH_DIR/hooks."""

    def __init__(self, *, actual: Path, expected: Path) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(
            f'hook at {actual} does not match CLAUDE_EXEC_LAUNCH_DIR/hooks ({expected}). '
            f'Hooks are wired to a different tree than CLAUDE_EXEC_LAUNCH_DIR points to.'
        )
