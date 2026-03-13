"""Claude Code hook input/output schemas.

See: https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pydantic

from cc_lib.schemas.base import CamelModel, StrictModel

# -- Hook inputs --------------------------------------------------------------


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
    permission_mode: str | None = None
    agent_type: str | None = None


class SessionEndHookInput(StrictModel):
    """SessionEnd hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionend
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['SessionEnd']
    reason: Literal['prompt_input_exit', 'clear', 'logout', 'bypass_permissions_disabled', 'other']
    permission_mode: str | None = None


class PreToolUseHookInput(StrictModel):
    """PreToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['PreToolUse']
    tool_name: str
    tool_input: Mapping[str, Any]  # strict_typing_linter.py: loose-typing — Claude Code sends arbitrary JSON per tool
    tool_use_id: str
    permission_mode: str | None = None


class PostToolUseHookInput(StrictModel):
    """PostToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#posttooluse
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['PostToolUse']
    tool_name: str
    tool_input: Mapping[str, Any]  # strict_typing_linter.py: loose-typing — Claude Code sends arbitrary JSON per tool
    tool_response: Any = None  # strict_typing_linter.py: loose-typing — response format varies by tool type
    tool_use_id: str
    permission_mode: str | None = None


# -- Tool input types ---------------------------------------------------------


class BashToolInput(StrictModel):
    """Bash tool input schema."""

    command: str
    description: str | None = None
    timeout: int | None = None
    run_in_background: bool | None = None
    dangerously_disable_sandbox: bool | None = pydantic.Field(default=None, alias='dangerouslyDisableSandbox')


class WriteToolInput(StrictModel):
    """Write tool input schema."""

    file_path: str
    content: str


class EditToolInput(StrictModel):
    """Edit tool input schema."""

    file_path: str
    old_string: str
    new_string: str
    replace_all: bool | None = None


# -- Hook outputs -------------------------------------------------------------


class PreToolUseDecision(CamelModel):
    """Permission decision within a PreToolUse hook output."""

    hook_event_name: Literal['PreToolUse'] = 'PreToolUse'
    permission_decision: Literal['allow', 'deny', 'ask']
    permission_decision_reason: str | None = None


class PreToolUseHookOutput(CamelModel):
    """PreToolUse hook output.

    Serialize with: ``model_dump_json(by_alias=True, exclude_none=True)``

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    hook_specific_output: PreToolUseDecision
