"""Exception hierarchy for cc_lib.

Two roots, orthogonal concerns — they don't share a base:

CCLibError                                # cc_lib's own failures
├── ClaudeContextError                    # session-context failures
│   ├── ClaudeProcessNotFoundError        # no Claude ancestor in process tree
│   ├── SessionNotFoundError              # session not in sessions.json
│   ├── InactiveSessionError              # session in sessions.json but state != 'active'
│   ├── MultipleActiveSessionsForPidError # multiple sessions claim the same claude_pid
│   └── MissingEnvVarError                # expected env var not set
├── HookTreeMismatchError                 # hook tree != CLAUDE_EXEC_LAUNCH_DIR/hooks
└── RivalSessionError                     # another live claude process owns this session_id

ResolvableError                           # workspace-shared base for known failure
                                          # modes with structured remediation. See class
                                          # docstring for the spec — it's the canonical
                                          # reference for the concept in the workspace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = [
    'CCLibError',
    'ClaudeContextError',
    'ClaudeProcessNotFoundError',
    'HookTreeMismatchError',
    'InactiveSessionError',
    'MissingEnvVarError',
    'MultipleActiveSessionsForPidError',
    'ResolvableError',
    'RivalSessionError',
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


class InactiveSessionError(ClaudeContextError):
    """Session exists in sessions.json but is not in 'active' state."""


class MultipleActiveSessionsForPidError(ClaudeContextError):
    """Multiple sessions in sessions.json claim the same active claude_pid.

    Workspace-state invariant violation — should not occur in normal operation.
    """


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


class RivalSessionError(CCLibError):
    """Another live claude process owns this session_id."""

    def __init__(self, *, session_id: str, rival_pid: int, claude_pid: int) -> None:
        self.session_id = session_id
        self.rival_pid = rival_pid
        self.claude_pid = claude_pid
        super().__init__(
            f'session_id {session_id} is owned by live pid {rival_pid} '
            f'(this process is pid {claude_pid}). Refusing to modify the entry.'
        )


class ResolvableError(Exception):
    """Workspace-shared base for known failure modes with structured remediation.

    Subclass when the failure is (a) recognizable — the pattern has a stable
    ``code`` — and (b) actionable — a sufficiently-equipped consumer can move
    past it via the inline ``suggestions``, the workflow at ``docs_url``, a
    source-side fix, or escalation. "Resolution" is broad: kill-and-retry, edit
    a schema, grant a macOS permission, re-run a patcher.

    Each instance encodes its own provenance:
        code      — identity:  which failure pattern this is
        docs_url  — history:   where it's documented in the workspace
        context   — locale:    the variables that situate this occurrence

    Consumers (human readers, agents in the loop, hook handlers, MCP clients,
    log pipelines) dispatch on the same structured fields however suits their
    medium. The class promises a wire shape, not a rendering.

    The motivating use case in this workspace is agent-loop engagement: a
    Claude Code session reading tool output can dispatch on ``code`` and fetch
    ``docs_url`` to refine its recovery attempt. The wire shape pays off most
    clearly when an LLM is in the loop, but the shape doesn't require one.

    Field shape converges with rustc Diagnostic, RFC 9457 Problem Details, and
    Pydantic v2 ``ValidationError.errors()``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        title: str | None = None,
        suggestions: Sequence[str] = (),
        docs_url: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.title = title
        self.suggestions = tuple(suggestions)
        self.docs_url = docs_url
        self.context: Mapping[str, str] = dict(context) if context else {}
