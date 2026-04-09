"""Claude Code hook input/output schemas.

See: https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

__all__ = [
    'BashToolInput',
    'EditToolInput',
    'PostToolUseHookInput',
    'PostToolUseHookOutput',
    'PostToolUseSpecificOutput',
    'PreToolUseHookInput',
    'PreToolUseHookOutput',
    'PreToolUseSpecificOutput',
    'SessionEndHookInput',
    'SessionStartHookInput',
    'SessionStartHookOutput',
    'SessionStartSpecificOutput',
    'SyncHookOutput',
    'UserPromptSubmitHookOutput',
    'UserPromptSubmitSpecificOutput',
    'WriteToolInput',
]

from collections.abc import Sequence
from typing import Literal

import pydantic
from pydantic import JsonValue

from cc_lib.schemas.base import CamelModel, StrictModel
from cc_lib.types import JsonObject

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
    agent_id: str | None = None
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
    agent_id: str | None = None
    agent_type: str | None = None


class PreToolUseHookInput(StrictModel):
    """PreToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['PreToolUse']
    tool_name: str
    tool_input: JsonObject
    tool_use_id: str
    permission_mode: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None


class PostToolUseHookInput(StrictModel):
    """PostToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#posttooluse
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['PostToolUse']
    tool_name: str
    tool_input: JsonObject
    tool_response: JsonValue | None = None
    tool_use_id: str
    permission_mode: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None


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


# -- Hook output envelope -----------------------------------------------------


class SyncHookOutput(CamelModel):
    """Shared top-level envelope for all hook JSON output.

    All fields are optional. Hooks return this structure on stdout as JSON.
    The ``hook_specific_output`` field is overridden by per-event output models.
    """

    continue_: bool | None = pydantic.Field(None, alias='continue')
    suppress_output: bool | None = None
    stop_reason: str | None = None
    decision: Literal['approve', 'block'] | None = None
    reason: str | None = None
    system_message: str | None = None


# -- Hook-specific outputs ----------------------------------------------------


class PreToolUseSpecificOutput(CamelModel):
    """PreToolUse hookSpecificOutput — permission decisions and input modification."""

    hook_event_name: Literal['PreToolUse'] = 'PreToolUse'
    permission_decision: Literal['allow', 'deny', 'ask', 'defer']
    permission_decision_reason: str | None = None
    updated_input: JsonObject | None = None
    additional_context: str | None = None


class PostToolUseSpecificOutput(CamelModel):
    """PostToolUse hookSpecificOutput — context injection and MCP output replacement."""

    hook_event_name: Literal['PostToolUse'] = 'PostToolUse'
    additional_context: str | None = None
    updated_mcp_tool_output: JsonValue | None = pydantic.Field(None, alias='updatedMCPToolOutput')


class UserPromptSubmitSpecificOutput(CamelModel):
    """UserPromptSubmit hookSpecificOutput — inject context before model responds."""

    hook_event_name: Literal['UserPromptSubmit'] = 'UserPromptSubmit'
    additional_context: str | None = None


class SessionStartSpecificOutput(CamelModel):
    """SessionStart hookSpecificOutput — context, initial message, and watch paths."""

    hook_event_name: Literal['SessionStart'] = 'SessionStart'
    additional_context: str | None = None
    initial_user_message: str | None = None
    watch_paths: Sequence[str] | None = None


# -- Composed hook outputs ----------------------------------------------------


class PreToolUseHookOutput(SyncHookOutput):
    """PreToolUse hook output.

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    hook_specific_output: PreToolUseSpecificOutput


class PostToolUseHookOutput(SyncHookOutput):
    """PostToolUse hook output.

    See: https://code.claude.com/docs/en/hooks#posttooluse
    """

    hook_specific_output: PostToolUseSpecificOutput | None = None


class UserPromptSubmitHookOutput(SyncHookOutput):
    """UserPromptSubmit hook output.

    See: https://code.claude.com/docs/en/hooks#userpromptsubmit
    """

    hook_specific_output: UserPromptSubmitSpecificOutput | None = None


class SessionStartHookOutput(SyncHookOutput):
    """SessionStart hook output.

    See: https://code.claude.com/docs/en/hooks#sessionstart
    """

    hook_specific_output: SessionStartSpecificOutput | None = None
