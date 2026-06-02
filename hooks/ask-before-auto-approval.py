#!/usr/bin/env -S uv run --quiet --script
"""Force a prompt before tools auto mode would silently auto-approve.

Auto mode bypasses ``permissions.ask``: the classifier routes only allow/deny
(https://github.com/anthropics/claude-code/issues/42797). This hook emits
``permissionDecision: "ask"`` under ``permission_mode == 'auto'`` for two cases:

  * AutoApprovalGate.ALWAYS — tools that are always a mutation/sensitive action, gated regardless
                              of arguments (built-in editors, non-GW MCP mutations, and
                              single-purpose google-workspace writes).
  * GoogleWorkspaceGate     — consolidated ``manage_*(action=…)`` tools, gated unless ``action``
                              is a known read (READ_ACTIONS). A write or unrecognized/missing
                              action is gated; only an explicit read passes. This stops a
                              destructive action from sailing through just because it was folded
                              into a larger multi-action tool, while still letting benign ``list``
                              reads auto-approve.

Other modes pass through:

  default           — native ``permissions.ask`` handles it
  acceptEdits       — auto-edits are the user's explicit choice
  plan              — read-only
  bypassPermissions — nuclear option; don't second-guess

Add an unconditional gate to ``AutoApprovalGate.ALWAYS`` (built-in names like ``Write``/``Edit``
or ``mcp__<server>__<tool>``); add a consolidated action-dispatch tool to
``GoogleWorkspaceGate.CONSOLIDATED``. Gating ``Bash`` matches every shell command — use
``permissions.ask`` patterns like ``Bash(rm -rf:*)`` for command-level control instead.

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
from collections.abc import Mapping

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import ClosedModel, SubsetModel
from cc_lib.schemas.hooks import (
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
)
from cc_lib.utils import get_claude_workspace_config_home_dir

GATE_DIR = get_claude_workspace_config_home_dir() / 'ask-before-auto-approval'

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> int:
    payload = PreToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    if payload.permission_mode != 'auto':
        return 0
    decision = AutoApprovalGate(payload.tool_name, payload.tool_input).decision
    if isinstance(decision, SkipPrompt):
        return 0
    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseSpecificOutput(
            permission_decision='ask',
            permission_decision_reason=decision.reason,
        ),
    )
    print(output.model_dump_json())
    return 0


class AutoApprovalGate:
    """Whether a tool call must be gated under auto mode.

    Checks the unconditional ``ALWAYS`` set first, then delegates consolidated
    ``manage_*(action=…)`` google-workspace tools to ``GoogleWorkspaceGate``.
    """

    # Tools that gate regardless of arguments: built-in editors, non-Google-Workspace MCP
    # mutations, and single-purpose google-workspace writes / sensitive actions.
    ALWAYS: frozenset[str] = frozenset(
        {
            'Edit',
            'Write',
            'mcp__google-workspace__append_table_rows',
            'mcp__google-workspace__batch_modify_gmail_message_labels',
            'mcp__google-workspace__batch_update_doc',
            'mcp__google-workspace__batch_update_form',
            'mcp__google-workspace__batch_update_presentation',
            'mcp__google-workspace__copy_drive_file',
            'mcp__google-workspace__create_calendar',
            'mcp__google-workspace__create_doc',
            'mcp__google-workspace__create_drive_file',
            'mcp__google-workspace__create_drive_folder',
            'mcp__google-workspace__create_form',
            'mcp__google-workspace__create_presentation',
            'mcp__google-workspace__create_reaction',
            'mcp__google-workspace__create_script_project',
            'mcp__google-workspace__create_sheet',
            'mcp__google-workspace__create_spreadsheet',
            'mcp__google-workspace__create_table_with_data',
            'mcp__google-workspace__create_version',
            'mcp__google-workspace__delete_calendar',
            'mcp__google-workspace__delete_script_project',
            'mcp__google-workspace__draft_gmail_message',
            'mcp__google-workspace__export_doc_to_pdf',
            'mcp__google-workspace__find_and_replace_doc',
            'mcp__google-workspace__format_sheet_range',
            'mcp__google-workspace__import_to_google_doc',
            'mcp__google-workspace__import_to_google_sheets',
            'mcp__google-workspace__import_to_google_slides',
            'mcp__google-workspace__insert_doc_elements',
            'mcp__google-workspace__insert_doc_image',
            'mcp__google-workspace__modify_doc_text',
            'mcp__google-workspace__modify_gmail_message_labels',
            'mcp__google-workspace__modify_sheet_values',
            'mcp__google-workspace__move_event',
            'mcp__google-workspace__move_sheet_rows',
            'mcp__google-workspace__resize_sheet_dimensions',
            'mcp__google-workspace__run_script_function',
            'mcp__google-workspace__send_gmail_message',
            'mcp__google-workspace__send_message',
            'mcp__google-workspace__set_drive_file_permissions',
            'mcp__google-workspace__set_publish_settings',
            'mcp__google-workspace__start_google_auth',
            'mcp__google-workspace__update_calendar',
            'mcp__google-workspace__update_doc_headers_footers',
            'mcp__google-workspace__update_drive_file',
            'mcp__google-workspace__update_paragraph_style',
            'mcp__google-workspace__update_script_content',
            'mcp__imessage-kit__send_message',
            'mcp__selenium-browser__navigate_with_profile_state',
        }
    )

    def __init__(self, tool_name: str, tool_input: Mapping[str, object]) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input

    @property
    def decision(self) -> PromptDecision:
        if self.tool_name in self.ALWAYS:
            return RequirePrompt.for_mutation(self.tool_name)
        if GoogleWorkspaceGate.handles(self.tool_name):
            return GoogleWorkspaceGate(self.tool_name, self.tool_input).decision
        return SkipPrompt()


class GoogleWorkspaceGate:
    """Gate one consolidated ``manage_*(action=…)`` google-workspace call.

    These dispatch on an ``action``: a read (``READ_ACTIONS``) auto-approves; a write or an
    unrecognized/missing action gates. Single-purpose google-workspace writes are unconditional
    and live in ``AutoApprovalGate.ALWAYS`` instead.
    """

    PREFIX = 'mcp__google-workspace__'

    CONSOLIDATED: frozenset[str] = frozenset(
        {
            'manage_calendar_sharing',
            'manage_conditional_formatting',
            'manage_contact',
            'manage_contact_group',
            'manage_contacts_batch',
            'manage_deployment',
            'manage_doc_tab',
            'manage_document_comment',
            'manage_drive_access',
            'manage_event',
            'manage_focus_time',
            'manage_gmail_filter',
            'manage_gmail_label',
            'manage_out_of_office',
            'manage_presentation_comment',
            'manage_spreadsheet_comment',
            'manage_task',
            'manage_task_list',
        }
    )

    # Read-only action verbs that may auto-approve. Only ``list`` occurs in the current toolset;
    # the rest are forward-looking.
    READ_ACTIONS: frozenset[str] = frozenset({'get', 'list', 'query', 'read', 'search'})

    def __init__(self, tool_name: str, tool_input: Mapping[str, object]) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input

    @classmethod
    def handles(cls, tool_name: str) -> bool:
        """True if ``tool_name`` is one of the consolidated google-workspace tools."""
        return tool_name.startswith(cls.PREFIX) and tool_name.removeprefix(cls.PREFIX) in cls.CONSOLIDATED

    @property
    def decision(self) -> PromptDecision:
        action = (self._action or '').strip().lower()
        if action in self.READ_ACTIONS:
            return SkipPrompt()
        return RequirePrompt.for_mutation(f'{self.tool_name}(action={action or "<unspecified>"})')

    @property
    def _action(self) -> str | None:
        """The ``action`` field (None when absent; a non-string action fails loud via the boundary)."""
        return GoogleWorkspaceToolInput.model_validate(self.tool_input).action


class GoogleWorkspaceToolInput(SubsetModel):
    """The single field ``GoogleWorkspaceGate`` reads from a google-workspace tool_input."""

    action: str | None = None
    """Dispatch action of a consolidated manage_* tool; None for single-purpose tools."""


class RequirePrompt(ClosedModel):
    """Auto mode must not silently auto-approve this call — require a manual prompt."""

    reason: str
    """Explanation surfaced in the approval prompt."""

    @classmethod
    def for_mutation(cls, subject: str) -> RequirePrompt:
        """Require a prompt for ``subject`` (a tool name or ``tool(action=…)``) that mutates."""
        return cls(reason=f'{subject} mutates under auto mode — manual approval required')


class SkipPrompt(ClosedModel):
    """No prompt needed; auto mode may auto-approve this call."""


type PromptDecision = RequirePrompt | SkipPrompt


if __name__ == '__main__':
    sys.exit(main())
