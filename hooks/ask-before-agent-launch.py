#!/usr/bin/env -S uv run --quiet --script
"""PreToolUse hook to gate privileged subagent launches.

Requires user approval before spawning a subagent with permissionMode:
bypassPermissions.

See: https://code.claude.com/docs/en/hooks#pretooluse
"""

# /// script
# dependencies = [
#   "cc_lib",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import json
import sys

GATED_AGENTS = {
    'code-review-validated',
    'unrestricted-worker',
}


def ask_stdlib(reason: str) -> None:
    """Emit an 'ask' decision using only stdlib (no pydantic required)."""
    json.dump(
        {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'ask',
                'permissionDecisionReason': reason,
            },
        },
        sys.stdout,
    )
    sys.exit(0)


try:
    import pydantic
    from cc_lib.schemas.hooks import (
        PreToolUseDecision,
        PreToolUseHookInput,
        PreToolUseHookOutput,
    )
except Exception as e:  # exception_safety_linter.py: swallowed-exception — fail closed: import error triggers manual approval, not silent pass
    # Fail CLOSED: if deps can't be imported, require manual approval
    print(f'hook import error: {e}', file=sys.stderr)
    ask_stdlib('hook import error — requesting manual approval')


def ask(reason: str) -> None:
    """Emit an 'ask' decision using Pydantic models."""
    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseDecision(
            permission_decision='ask',
            permission_decision_reason=reason,
        ),
    )
    print(output.model_dump_json())
    sys.exit(0)


try:
    hook_data = PreToolUseHookInput.model_validate_json(sys.stdin.read())
except pydantic.ValidationError as e:
    # Fail CLOSED: if we can't parse the input, require manual approval
    print(f'hook validation error: {e}', file=sys.stderr)
    ask('hook validation error — requesting manual approval')

subagent_type = hook_data.tool_input.get('subagent_type', '')
if subagent_type in GATED_AGENTS:
    ask(f'{subagent_type} uses bypassPermissions mode')
