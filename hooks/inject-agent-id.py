#!/usr/bin/env -S uv run --quiet --script
"""PreToolUse hook: inject CLAUDE_CODE_AGENT_ID into Bash subprocess environments.

Sub-agents spawned by Claude Code's Agent tool have unique agent IDs
(e.g., a3f8c1b4d5e6f7a8), but these are internal to Claude Code's Node.js
process — they never reach Bash subprocesses. This hook bridges that gap.

When a Bash tool call originates from a sub-agent, the hook rewrites the
command to prefix it with ``export CLAUDE_CODE_AGENT_ID=<agent_id>;``,
making the ID available as an environment variable for the entire command,
including compound commands (&&, ||, ;, |), time, sudo, and subshells.

Main-thread commands (agent_id is None) pass through unmodified. The
absence of CLAUDE_CODE_AGENT_ID in the environment implies main thread.

Uses the updatedInput mechanism (no permissionDecision) so normal
permission rules still apply — the user sees the modified command in the
approval prompt.

Related issues:
    https://github.com/anthropics/claude-code/issues/29068
        agent_id added to all hook events (shipped v2.1.64)
    https://github.com/anthropics/claude-code/issues/35447
        Request for CLAUDE_AGENT_ID as native env var
    https://github.com/anthropics/claude-code/issues/32514
        Request for agent_context in MCP tool calls
    https://github.com/anthropics/claude-code/issues/14859
        Agent hierarchy in hook events

Caveats:
    https://github.com/anthropics/claude-code/issues/15897
        If multiple PreToolUse hooks return updatedInput for the same tool
        call, only the first one's updatedInput survives. Place this hook
        first in the hooks array if other Bash-matching hooks exist.
    https://github.com/anthropics/claude-code/issues/34692
        Hooks may not fire for sub-agent tool calls in --bare mode.
        Normal interactive sessions are unaffected.

See: https://code.claude.com/docs/en/hooks#pretooluse
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import sys

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import (
    BashToolInput,
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
)

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> None:
    hook_data = PreToolUseHookInput.model_validate_json(sys.stdin.read())

    if hook_data.tool_name != 'Bash':
        raise RuntimeError(
            f'inject-agent-id: expected Bash tool, got {hook_data.tool_name!r} — check matcher config in settings.json'
        )

    if not hook_data.agent_id:
        return  # Main thread — no injection needed

    bash_input = BashToolInput.model_validate(hook_data.tool_input)
    if not bash_input.command:
        return

    # Use `export VAR=val;` prefix — sets the variable in the shell environment
    # for the entire command, including compound commands (&&, ||, ;, |), time,
    # sudo, and subshells. Empirically verified: `env VAR=val` and inline
    # `VAR=val` only apply to the first simple command in a compound chain.
    modified_command = f'export CLAUDE_CODE_AGENT_ID={hook_data.agent_id}; {bash_input.command}'

    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseSpecificOutput(
            updated_input={'command': modified_command},
        ),
    )
    print(output.model_dump_json(by_alias=True, exclude_none=True))


@boundary.handler(Exception)
def _handle_error(exc: Exception) -> None:
    print(f'inject-agent-id hook error: {exc!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
