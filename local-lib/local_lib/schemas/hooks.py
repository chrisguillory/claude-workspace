"""Claude Code hook input/output schemas.

See: https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

from typing import Any, Literal

import pydantic


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation."""

    model_config = pydantic.ConfigDict(
        extra='forbid',  # Reject unknown fields (fail-fast)
        strict=True,  # Strict type coercion
        frozen=True,  # Immutable after creation
    )


class SessionStartHookInput(StrictModel):
    """SessionStart hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionstart
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['SessionStart']
    source: Literal['startup', 'resume', 'compact', 'clear']
    model: str | None = None


class SessionEndHookInput(StrictModel):
    """SessionEnd hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionend
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['SessionEnd']
    reason: Literal['prompt_input_exit', 'clear', 'logout', 'other']


# --- PreToolUse hook types ---


class PreToolUseHookInput(StrictModel):
    """PreToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['PreToolUse']
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    permission_mode: str | None = None


class PreToolUseHookOutput(StrictModel):
    """PreToolUse hook output — serialize with model_dump_json(by_alias=True).

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    model_config = pydantic.ConfigDict(populate_by_name=True)

    hook_specific_output: PreToolUseDecision = pydantic.Field(alias='hookSpecificOutput')


class PreToolUseDecision(StrictModel):
    """Permission decision within a PreToolUse hook output."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    hook_event_name: Literal['PreToolUse'] = pydantic.Field(default='PreToolUse', alias='hookEventName')
    permission_decision: Literal['allow', 'deny', 'ask'] = pydantic.Field(alias='permissionDecision')
    permission_decision_reason: str | None = pydantic.Field(default=None, alias='permissionDecisionReason')
