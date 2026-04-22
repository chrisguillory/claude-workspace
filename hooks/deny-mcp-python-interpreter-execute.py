#!/usr/bin/env -S uv run --quiet --no-project --script
"""PreToolUse hook: deny ``mcp__python-interpreter__execute`` with CLI guidance.

Native ``permissions.deny`` is a flat list of string patterns — no per-rule
reason surfaced to Claude. This hook fills the gap for a single tool: it emits
``permissionDecision: "deny"`` with a ``permissionDecisionReason`` pointing at
the CLI alternative. Per the hook docs, reason on deny is shown to Claude
(unlike allow/ask where it's user-only), so the first denied call nudges Claude
to pivot rather than guess.

Why deny this specific tool: approval prompts for the MCP variant display
escaped-JSON code, which is unreadable for multi-line scripts. The
``python-interpreter`` CLI is feature-identical (same persistent scope, same
auto-install via uv) but takes heredoc input that renders as plain text in
approval dialogs. See ``~/claude-workspace/CLAUDE.md:749-755`` for the shape.

Scoping: this hook assumes it is only invoked for its target tool. Pin the
matcher in ``~/.claude/settings.json`` so it doesn't run for anything else::

    {
      "matcher": "^mcp__python-interpreter__execute$",
      "hooks": [{
        "type": "command",
        "command": "/Users/chris/claude-workspace/hooks/deny-mcp-python-interpreter-execute.py"
      }]
    }

Hook docs: https://code.claude.com/docs/en/hooks#pretooluse
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///

from __future__ import annotations

import sys

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import (
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
)

boundary = ErrorBoundary(exit_code=0)


@boundary
def main() -> None:
    """Deny the target tool with CLI-alternative guidance."""
    PreToolUseHookInput.model_validate_json(sys.stdin.read())

    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseSpecificOutput(
            permission_decision='deny',
            permission_decision_reason=DENIAL_REASON,
        ),
    )
    print(output.model_dump_json())


DENIAL_REASON = """\
Use the `python-interpreter` CLI instead. It speaks to the same MCP server over \
a Unix socket, so persistent scope, auto-install of packages, and registered \
interpreters all behave identically — but approval prompts show readable heredoc \
text instead of escaped JSON.

    python-interpreter <<'PY'
    print("hello")
    PY

Reference: ~/claude-workspace/CLAUDE.md:749-755.
"""


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'deny-mcp-python-interpreter-execute hook error: {exc!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
