"""Claude Code hook input/output schemas.

Attribute docstrings are taken verbatim from Anthropic's zod `.describe(...)`
calls in the Claude Code binary — the authoritative source for each field's
meaning. Pydantic exposes these via `use_attribute_docstrings=True` on the
base models.

See: https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

__all__ = [
    'AddDirectoriesSuggestion',
    'AddRulesSuggestion',
    'BashToolInput',
    'CompactTrigger',
    'ConfigChangeHookInput',
    'ConfigChangeSource',
    'CwdChangedHookInput',
    'EditToolInput',
    'ElicitationHookInput',
    'ElicitationMode',
    'ElicitationResultAction',
    'ElicitationResultHookInput',
    'FileChangedEvent',
    'FileChangedHookInput',
    'HookEffort',
    'InstructionsLoadedHookInput',
    'InstructionsLoadedLoadReason',
    'InstructionsLoadedMemoryType',
    'NotificationHookInput',
    'PermissionDeniedHookInput',
    'PermissionDestination',
    'PermissionMode',
    'PermissionRequestHookInput',
    'PermissionRule',
    'PermissionRuleBehavior',
    'PermissionSuggestion',
    'PostCompactHookInput',
    'PostToolBatchHookInput',
    'PostToolBatchToolCall',
    'PostToolUseFailureHookInput',
    'PostToolUseHookInput',
    'PostToolUseHookOutput',
    'PostToolUseSpecificOutput',
    'PreCompactHookInput',
    'PreToolUseHookInput',
    'PreToolUseHookOutput',
    'PreToolUseSpecificOutput',
    'RemoveDirectoriesSuggestion',
    'RemoveRulesSuggestion',
    'ReplaceRulesSuggestion',
    'SessionEndHookInput',
    'SessionEndReason',
    'SessionStartHookInput',
    'SessionStartHookOutput',
    'SessionStartSource',
    'SessionStartSpecificOutput',
    'SetModeSuggestion',
    'SetupHookInput',
    'SetupTrigger',
    'StopFailureError',
    'StopFailureHookInput',
    'StopHookInput',
    'SubagentStartHookInput',
    'SubagentStopHookInput',
    'SyncHookOutput',
    'TaskCompletedHookInput',
    'TaskCreatedHookInput',
    'TeammateIdleHookInput',
    'UserPromptExpansionHookInput',
    'UserPromptExpansionType',
    'UserPromptSubmitHookInput',
    'UserPromptSubmitHookOutput',
    'UserPromptSubmitSpecificOutput',
    'WorktreeCreateHookInput',
    'WorktreeRemoveHookInput',
    'WriteToolInput',
]

from collections.abc import Sequence
from typing import Annotated, Literal

import pydantic
from pydantic import JsonValue

from cc_lib.schemas.base import CamelModel, StrictModel
from cc_lib.types import EffortLevel, JsonObject

# -- Shared field types -------------------------------------------------------

# Type aliases (alphabetical)

type CompactTrigger = Literal['manual', 'auto']

type ConfigChangeSource = Literal[
    'user_settings',
    'project_settings',
    'local_settings',
    'policy_settings',
    'skills',
]

type ElicitationMode = Literal['form', 'url']

type ElicitationResultAction = Literal['accept', 'decline', 'cancel']

type FileChangedEvent = Literal['change', 'add', 'unlink']

type InstructionsLoadedLoadReason = Literal[
    'session_start',
    'nested_traversal',
    'path_glob_match',
    'include',
    'compact',
]

type InstructionsLoadedMemoryType = Literal['User', 'Project', 'Local', 'Managed']

type PermissionDestination = Literal['userSettings', 'projectSettings', 'localSettings', 'session', 'cliArg']

type PermissionMode = Literal[
    'default',
    'acceptEdits',
    'plan',
    'auto',
    'dontAsk',
    'bypassPermissions',
]

type PermissionRuleBehavior = Literal['allow', 'deny', 'ask']

type SessionEndReason = Literal[
    'clear',
    'resume',
    'logout',
    'prompt_input_exit',
    'other',
    'bypass_permissions_disabled',
]

type SessionStartSource = Literal['startup', 'resume', 'clear', 'compact']

type SetupTrigger = Literal['init', 'maintenance']

type StopFailureError = Literal[
    'authentication_failed',
    'oauth_org_not_allowed',
    'billing_error',
    'rate_limit',
    'invalid_request',
    'server_error',
    'unknown',
    'max_output_tokens',
]

type UserPromptExpansionType = Literal['slash_command', 'mcp_prompt']


# Sub-models (referenced by hook inputs)


class HookEffort(StrictModel):
    """Reasoning effort applied to a hook's turn."""

    level: EffortLevel
    """
    Active effort level for the current turn (e.g., "low", "medium", "high", "xhigh", "max"),
    after any silent downgrade for the selected model. Also exposed to hook commands and Bash
    as the CLAUDE_EFFORT env var.
    """


class PermissionRule(CamelModel):
    """A single tool-name + optional rule-content entry in a permission suggestion."""

    tool_name: str
    rule_content: str | None = None


class PostToolBatchToolCall(StrictModel):
    """Single tool-use record within a PostToolBatch invocation."""

    tool_name: str
    tool_input: JsonObject
    tool_use_id: str
    tool_response: JsonValue | None = None


# Discriminated union: PermissionRequestHookInput.permission_suggestions[] —
# six variants tagged by the `type` field.


class AddRulesSuggestion(StrictModel):
    """Permission suggestion: append rules to the destination."""

    type: Literal['addRules']
    rules: Sequence[PermissionRule]
    behavior: PermissionRuleBehavior
    destination: PermissionDestination


class ReplaceRulesSuggestion(StrictModel):
    """Permission suggestion: replace all rules in the destination."""

    type: Literal['replaceRules']
    rules: Sequence[PermissionRule]
    behavior: PermissionRuleBehavior
    destination: PermissionDestination


class RemoveRulesSuggestion(StrictModel):
    """Permission suggestion: remove the listed rules from the destination."""

    type: Literal['removeRules']
    rules: Sequence[PermissionRule]
    behavior: PermissionRuleBehavior
    destination: PermissionDestination


class SetModeSuggestion(StrictModel):
    """Permission suggestion: set the permission mode at the destination."""

    type: Literal['setMode']
    mode: PermissionMode
    destination: PermissionDestination


class AddDirectoriesSuggestion(StrictModel):
    """Permission suggestion: allow access to additional directories."""

    type: Literal['addDirectories']
    directories: Sequence[str]
    destination: PermissionDestination


class RemoveDirectoriesSuggestion(StrictModel):
    """Permission suggestion: revoke directory access at the destination."""

    type: Literal['removeDirectories']
    directories: Sequence[str]
    destination: PermissionDestination


type PermissionSuggestion = Annotated[
    AddRulesSuggestion
    | ReplaceRulesSuggestion
    | RemoveRulesSuggestion
    | SetModeSuggestion
    | AddDirectoriesSuggestion
    | RemoveDirectoriesSuggestion,
    pydantic.Field(discriminator='type'),
]


# -- Hook inputs --------------------------------------------------------------


class _HookInputBase(StrictModel):
    """Fields shared by every hook input (Anthropic's ``cz()`` composition base).

    Every hook schema in the Claude Code binary is defined as
    ``cz().and(h.object({hook_event_name: ..., ...specific fields...}))``.
    This class mirrors that composition so per-hook subclasses only declare
    what's unique to them.
    """

    session_id: str
    cwd: str
    transcript_path: str
    permission_mode: PermissionMode | None = None
    agent_id: str | None = None
    """Subagent identifier.

    Present only when the hook fires from within a subagent (e.g., a tool called
    by an AgentTool worker). Absent for the main thread, even in --agent
    sessions. Use this field (not agent_type) to distinguish subagent calls from
    main-thread calls.
    """
    agent_type: str | None = None
    """Agent type name (e.g., "general-purpose", "code-reviewer").

    Present when the hook fires from within a subagent (alongside agent_id), or
    on the main thread of a session started with --agent (without agent_id).
    """
    effort: HookEffort | None = None
    """
    Reasoning effort applied to the current turn. Same shape as StatusLineCommandInput.effort.
    Present for hooks that fire within a tool-use context (PreToolUse, PostToolUse, Stop,
    SubagentStop, etc.) on a model that supports the effort parameter; absent for
    session-lifecycle hooks and models without effort support.
    """


# Session lifecycle


class SessionStartHookInput(_HookInputBase):
    """SessionStart hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionstart
    """

    hook_event_name: Literal['SessionStart']
    source: SessionStartSource
    model: str | None = None


class SessionEndHookInput(_HookInputBase):
    """SessionEnd hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionend
    """

    hook_event_name: Literal['SessionEnd']
    reason: SessionEndReason


class SetupHookInput(_HookInputBase):
    """Setup hook input schema.

    See: https://code.claude.com/docs/en/hooks#setup
    """

    hook_event_name: Literal['Setup']
    trigger: SetupTrigger


# Compact lifecycle


class PreCompactHookInput(_HookInputBase):
    """PreCompact hook input schema.

    See: https://code.claude.com/docs/en/hooks#precompact
    """

    hook_event_name: Literal['PreCompact']
    trigger: CompactTrigger
    custom_instructions: str | None


class PostCompactHookInput(_HookInputBase):
    """PostCompact hook input schema.

    See: https://code.claude.com/docs/en/hooks#postcompact
    """

    hook_event_name: Literal['PostCompact']
    trigger: CompactTrigger
    compact_summary: str
    """The conversation summary produced by compaction."""


# Tool-use lifecycle


class PreToolUseHookInput(_HookInputBase):
    """PreToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#pretooluse
    """

    hook_event_name: Literal['PreToolUse']
    tool_name: str
    tool_input: JsonObject
    tool_use_id: str


class PermissionRequestHookInput(_HookInputBase):
    """PermissionRequest hook input schema.

    See: https://code.claude.com/docs/en/hooks#permissionrequest
    """

    hook_event_name: Literal['PermissionRequest']
    tool_name: str
    tool_input: JsonObject
    permission_suggestions: Sequence[PermissionSuggestion] | None = None


class PermissionDeniedHookInput(_HookInputBase):
    """PermissionDenied hook input schema.

    See: https://code.claude.com/docs/en/hooks#permissiondenied
    """

    hook_event_name: Literal['PermissionDenied']
    tool_name: str
    tool_input: JsonObject
    tool_use_id: str
    reason: str


class PostToolUseHookInput(_HookInputBase):
    """PostToolUse hook input schema.

    See: https://code.claude.com/docs/en/hooks#posttooluse
    """

    hook_event_name: Literal['PostToolUse']
    tool_name: str
    tool_input: JsonObject
    tool_response: JsonValue | None = None
    tool_use_id: str
    duration_ms: int | None = None
    """Tool execution time in milliseconds.

    Excludes permission-prompt and hook time.
    """


class PostToolUseFailureHookInput(_HookInputBase):
    """PostToolUseFailure hook input schema.

    See: https://code.claude.com/docs/en/hooks#posttoolusefailure
    """

    hook_event_name: Literal['PostToolUseFailure']
    tool_name: str
    tool_input: JsonObject
    tool_use_id: str
    error: str
    is_interrupt: bool | None = None
    duration_ms: int | None = None
    """Tool execution time in milliseconds.

    Excludes permission-prompt and hook time.
    """


class PostToolBatchHookInput(_HookInputBase):
    """PostToolBatch hook input schema.

    Fired once after every tool call in a batch has resolved, before the next model
    request. PostToolUse fires per-tool and may run concurrently for parallel tool calls;
    PostToolBatch fires exactly once with the full batch.

    See: https://code.claude.com/docs/en/hooks#posttoolbatch
    """

    hook_event_name: Literal['PostToolBatch']
    tool_calls: Sequence[PostToolBatchToolCall]


# Stop lifecycle


class StopHookInput(_HookInputBase):
    """Stop hook input schema.

    See: https://code.claude.com/docs/en/hooks#stop
    """

    hook_event_name: Literal['Stop']
    stop_hook_active: bool
    last_assistant_message: str | None = None
    """Text content of the last assistant message before stopping.

    Avoids the need to read and parse the transcript file.
    """


class StopFailureHookInput(_HookInputBase):
    """StopFailure hook input schema.

    See: https://code.claude.com/docs/en/hooks#stopfailure
    """

    hook_event_name: Literal['StopFailure']
    error: StopFailureError
    error_details: str | None = None
    last_assistant_message: str | None = None


# Subagent lifecycle


class SubagentStartHookInput(_HookInputBase):
    """SubagentStart hook input schema.

    See: https://code.claude.com/docs/en/hooks#subagentstart
    """

    hook_event_name: Literal['SubagentStart']
    agent_id: str
    agent_type: str


class SubagentStopHookInput(_HookInputBase):
    """SubagentStop hook input schema.

    See: https://code.claude.com/docs/en/hooks#subagentstop
    """

    hook_event_name: Literal['SubagentStop']
    stop_hook_active: bool
    agent_id: str
    agent_transcript_path: str
    agent_type: str
    last_assistant_message: str | None = None
    """Text content of the last assistant message before stopping.

    Avoids the need to read and parse the transcript file.
    """


# User interaction


class UserPromptSubmitHookInput(_HookInputBase):
    """UserPromptSubmit hook input schema.

    See: https://code.claude.com/docs/en/hooks#userpromptsubmit
    """

    hook_event_name: Literal['UserPromptSubmit']
    prompt: str
    session_title: str | None = None


class UserPromptExpansionHookInput(_HookInputBase):
    """UserPromptExpansion hook input schema.

    See: https://code.claude.com/docs/en/hooks#userpromptexpansion
    """

    hook_event_name: Literal['UserPromptExpansion']
    expansion_type: UserPromptExpansionType
    command_name: str
    command_args: str
    command_source: str | None = None
    prompt: str


class NotificationHookInput(_HookInputBase):
    """Notification hook input schema.

    See: https://code.claude.com/docs/en/hooks#notification
    """

    hook_event_name: Literal['Notification']
    message: str
    title: str | None = None
    notification_type: str


class ElicitationHookInput(_HookInputBase):
    """Elicitation hook input schema.

    Fired when an MCP server requests user input. Hooks can auto-respond
    (accept/decline) instead of showing the dialog.

    See: https://code.claude.com/docs/en/hooks#elicitation
    """

    hook_event_name: Literal['Elicitation']
    mcp_server_name: str
    message: str
    mode: ElicitationMode | None = None
    url: str | None = None
    elicitation_id: str | None = None
    requested_schema: JsonObject | None = None


class ElicitationResultHookInput(_HookInputBase):
    """ElicitationResult hook input schema.

    Fired after the user responds to an MCP elicitation. Hooks can observe or
    override the response before it is sent to the server.

    See: https://code.claude.com/docs/en/hooks#elicitationresult
    """

    hook_event_name: Literal['ElicitationResult']
    mcp_server_name: str
    elicitation_id: str | None = None
    mode: ElicitationMode | None = None
    action: ElicitationResultAction
    content: JsonObject | None = None


# Configuration / state changes


class ConfigChangeHookInput(_HookInputBase):
    """ConfigChange hook input schema.

    See: https://code.claude.com/docs/en/hooks#configchange
    """

    hook_event_name: Literal['ConfigChange']
    source: ConfigChangeSource
    file_path: str | None = None


class InstructionsLoadedHookInput(_HookInputBase):
    """InstructionsLoaded hook input schema.

    See: https://code.claude.com/docs/en/hooks#instructionsloaded
    """

    hook_event_name: Literal['InstructionsLoaded']
    file_path: str
    memory_type: InstructionsLoadedMemoryType
    load_reason: InstructionsLoadedLoadReason
    globs: Sequence[str] | None = None
    trigger_file_path: str | None = None
    parent_file_path: str | None = None


class CwdChangedHookInput(_HookInputBase):
    """CwdChanged hook input schema.

    See: https://code.claude.com/docs/en/hooks#cwdchanged
    """

    hook_event_name: Literal['CwdChanged']
    old_cwd: str
    new_cwd: str


class FileChangedHookInput(_HookInputBase):
    """FileChanged hook input schema.

    See: https://code.claude.com/docs/en/hooks#filechanged
    """

    hook_event_name: Literal['FileChanged']
    file_path: str
    event: FileChangedEvent


# Tasks / teammates


class TaskCreatedHookInput(_HookInputBase):
    """TaskCreated hook input schema.

    See: https://code.claude.com/docs/en/hooks#taskcreated
    """

    hook_event_name: Literal['TaskCreated']
    task_id: str
    task_subject: str
    task_description: str | None = None
    teammate_name: str | None = None
    team_name: str | None = None


class TaskCompletedHookInput(_HookInputBase):
    """TaskCompleted hook input schema.

    See: https://code.claude.com/docs/en/hooks#taskcompleted
    """

    hook_event_name: Literal['TaskCompleted']
    task_id: str
    task_subject: str
    task_description: str | None = None
    teammate_name: str | None = None
    team_name: str | None = None


class TeammateIdleHookInput(_HookInputBase):
    """TeammateIdle hook input schema.

    See: https://code.claude.com/docs/en/hooks#teammateidle
    """

    hook_event_name: Literal['TeammateIdle']
    teammate_name: str
    team_name: str


# Worktree


class WorktreeCreateHookInput(_HookInputBase):
    """WorktreeCreate hook input schema.

    See: https://code.claude.com/docs/en/hooks#worktreecreate
    """

    hook_event_name: Literal['WorktreeCreate']
    name: str


class WorktreeRemoveHookInput(_HookInputBase):
    """WorktreeRemove hook input schema.

    See: https://code.claude.com/docs/en/hooks#worktreeremove
    """

    hook_event_name: Literal['WorktreeRemove']
    worktree_path: str


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
    permission_decision: Literal['allow', 'deny', 'ask', 'defer'] | None = None
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
    session_title: str | None = None
    suppress_original_prompt: bool | None = None
    """When decision is "block", omit the original prompt from the block message."""


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
