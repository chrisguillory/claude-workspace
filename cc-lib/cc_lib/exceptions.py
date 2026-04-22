from __future__ import annotations

from pathlib import Path

__all__ = [
    'ClaudeProcessError',
    'HookTreeMismatchError',
]


class ClaudeProcessError(Exception):
    """Claude Code process not found or session not resolvable."""


class HookTreeMismatchError(Exception):
    """Hook script's location does not match CLAUDE_EXEC_LAUNCH_DIR/hooks."""

    def __init__(self, *, actual: Path, expected: Path) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(
            f'hook at {actual} does not match CLAUDE_EXEC_LAUNCH_DIR/hooks ({expected}). '
            f'Hooks are wired to a different tree than CLAUDE_EXEC_LAUNCH_DIR points to.'
        )
