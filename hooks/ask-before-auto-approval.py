#!/usr/bin/env -S uv run --quiet --script
"""Force a prompt before tools auto mode would silently auto-approve.

Auto mode bypasses ``permissions.ask``: the classifier routes only
allow/deny (https://github.com/anthropics/claude-code/issues/42797). This
hook emits ``permissionDecision: "ask"`` for tools in GATED_TOOLS when
``permission_mode == 'auto'``. Other modes pass through:

  default           — native ``permissions.ask`` handles it
  acceptEdits       — auto-edits are the user's explicit choice
  plan              — read-only
  bypassPermissions — nuclear option; don't second-guess

Extend GATED_TOOLS with built-in names (``Write``, ``Edit``, ``NotebookEdit``)
or MCP names (``mcp__<server>__<tool>``). Gating ``Bash`` matches every
shell command — use ``permissions.ask`` patterns like ``Bash(rm -rf:*)``
for command-level control instead.

Per-session bypass: touch
``~/.claude-workspace/ask-before-auto-approval/disabled-<session_id>``
(session id is ``$CLAUDE_CODE_SESSION_ID`` in any Bash call).

See: https://code.claude.com/docs/en/hooks#pretooluse
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import (
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
)

GATED_TOOLS = {
    'mcp__google-workspace__create_event',
    'Edit',
}

GATE_DIR = Path.home() / '.claude-workspace' / 'ask-before-auto-approval'

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> int:
    payload = PreToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    if payload.permission_mode != 'auto' or payload.tool_name not in GATED_TOOLS:
        return 0

    if (GATE_DIR / f'disabled-{payload.session_id}').exists():
        return 0

    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseSpecificOutput(
            permission_decision='ask',
            permission_decision_reason=f'{payload.tool_name} is gated under auto mode — manual approval required',
        ),
    )
    print(output.model_dump_json())
    return 0


if __name__ == '__main__':
    sys.exit(main())
