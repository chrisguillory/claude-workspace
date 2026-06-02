#!/usr/bin/env -S uv run --quiet --script
"""Force a prompt before tools auto mode would silently auto-approve.

Auto mode bypasses ``permissions.ask``: the classifier routes only allow/deny
(https://github.com/anthropics/claude-code/issues/42797). This hook emits
``permissionDecision: "ask"`` under ``permission_mode == 'auto'`` for two cases:

  * GATED_TOOLS        — tools that are always a mutation/sensitive action.
  * ACTION_GATED_TOOLS — consolidated ``manage_*(action=…)`` tools, gated unless
                         ``action`` is a known read (READONLY_ACTIONS). A write or
                         unrecognized/missing action is gated; only an explicit read
                         passes. This stops a destructive action from sailing through
                         just because it was folded into a larger multi-action tool,
                         while still letting benign ``list`` reads auto-approve.

Other modes pass through:

  default           — native ``permissions.ask`` handles it
  acceptEdits       — auto-edits are the user's explicit choice
  plan              — read-only
  bypassPermissions — nuclear option; don't second-guess

Extend GATED_TOOLS with built-in names (``Write``, ``Edit``, ``NotebookEdit``) or
MCP names (``mcp__<server>__<tool>``). For a consolidated tool whose action
vocabulary mixes reads and writes, add it to ACTION_GATED_TOOLS instead. Gating
``Bash`` matches every shell command — use ``permissions.ask`` patterns like
``Bash(rm -rf:*)`` for command-level control instead.

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
from cc_lib.schemas.hooks import (
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
)
from cc_lib.utils import get_claude_workspace_config_home_dir

# Always gated under auto mode: every invocation mutates or is sensitive.
GATED_TOOLS = {
    # Editing
    'Edit',
    'Write',
    # Selenium profile-state (sensitive auth import)
    'mcp__selenium-browser__navigate_with_profile_state',
    # iMessage mutations (outgoing messages to external parties)
    'mcp__imessage-kit__send_message',
    # Google Workspace — auth bootstrap (sensitive: initiates an OAuth flow)
    'mcp__google-workspace__start_google_auth',
    # Google Workspace — Calendar mutations (single-purpose)
    'mcp__google-workspace__create_calendar',
    'mcp__google-workspace__update_calendar',
    'mcp__google-workspace__delete_calendar',
    'mcp__google-workspace__move_event',
    # Google Workspace — Gmail mutations (single-purpose)
    'mcp__google-workspace__send_gmail_message',
    'mcp__google-workspace__draft_gmail_message',
    'mcp__google-workspace__modify_gmail_message_labels',
    'mcp__google-workspace__batch_modify_gmail_message_labels',
    # Google Workspace — Drive mutations (single-purpose)
    'mcp__google-workspace__create_drive_file',
    'mcp__google-workspace__create_drive_folder',
    'mcp__google-workspace__update_drive_file',
    'mcp__google-workspace__copy_drive_file',
    'mcp__google-workspace__set_drive_file_permissions',
    # Google Workspace — Docs mutations (single-purpose)
    'mcp__google-workspace__create_doc',
    'mcp__google-workspace__batch_update_doc',
    'mcp__google-workspace__modify_doc_text',
    'mcp__google-workspace__find_and_replace_doc',
    'mcp__google-workspace__insert_doc_elements',
    'mcp__google-workspace__insert_doc_image',
    'mcp__google-workspace__create_table_with_data',
    'mcp__google-workspace__update_paragraph_style',
    'mcp__google-workspace__update_doc_headers_footers',
    # Google Workspace — imports (create a Doc/Sheet/Slides from an uploaded file)
    'mcp__google-workspace__import_to_google_doc',
    'mcp__google-workspace__import_to_google_sheets',
    'mcp__google-workspace__import_to_google_slides',
    'mcp__google-workspace__export_doc_to_pdf',
    # Google Workspace — Sheets mutations (single-purpose)
    'mcp__google-workspace__create_spreadsheet',
    'mcp__google-workspace__create_sheet',
    'mcp__google-workspace__modify_sheet_values',
    'mcp__google-workspace__append_table_rows',
    'mcp__google-workspace__format_sheet_range',
    'mcp__google-workspace__resize_sheet_dimensions',
    'mcp__google-workspace__move_sheet_rows',
    # Google Workspace — Slides mutations (single-purpose)
    'mcp__google-workspace__create_presentation',
    'mcp__google-workspace__batch_update_presentation',
    # Google Workspace — Forms mutations (single-purpose)
    'mcp__google-workspace__create_form',
    'mcp__google-workspace__batch_update_form',
    'mcp__google-workspace__set_publish_settings',
    # Google Workspace — Chat mutations (single-purpose)
    'mcp__google-workspace__send_message',
    'mcp__google-workspace__create_reaction',
    # Google Workspace — Apps Script mutations (single-purpose)
    'mcp__google-workspace__create_script_project',
    'mcp__google-workspace__delete_script_project',
    'mcp__google-workspace__update_script_content',
    'mcp__google-workspace__create_version',
    'mcp__google-workspace__run_script_function',
}

# Consolidated ``manage_*(action=…)`` tools. Gated unless ``action`` is a known
# read. Among these tools, ``list`` is the only read action; every other action
# mutates (create/update/delete/add/remove/grant/revoke/move/rsvp/reply/resolve/
# rename/transfer_owner/modify_members/grant_batch/clear_completed/...). A missing
# or unrecognized action is treated as a write and gated (fail safe).
ACTION_GATED_TOOLS = {
    'mcp__google-workspace__manage_event',
    'mcp__google-workspace__manage_calendar_sharing',
    'mcp__google-workspace__manage_focus_time',
    'mcp__google-workspace__manage_out_of_office',
    'mcp__google-workspace__manage_gmail_filter',
    'mcp__google-workspace__manage_gmail_label',
    'mcp__google-workspace__manage_drive_access',
    'mcp__google-workspace__manage_doc_tab',
    'mcp__google-workspace__manage_document_comment',
    'mcp__google-workspace__manage_spreadsheet_comment',
    'mcp__google-workspace__manage_presentation_comment',
    'mcp__google-workspace__manage_conditional_formatting',
    'mcp__google-workspace__manage_contact',
    'mcp__google-workspace__manage_contact_group',
    'mcp__google-workspace__manage_contacts_batch',
    'mcp__google-workspace__manage_task',
    'mcp__google-workspace__manage_task_list',
    'mcp__google-workspace__manage_deployment',
}

# Read-only action verbs that may auto-approve inside an ACTION_GATED_TOOLS tool.
# Only ``list`` occurs in the current toolset; the rest are forward-looking.
READONLY_ACTIONS = {'list', 'get', 'read', 'search', 'query'}

GATE_DIR = get_claude_workspace_config_home_dir() / 'ask-before-auto-approval'

boundary = ErrorBoundary(exit_code=2)


def _gate_reason(tool_name: str, tool_input: Mapping[str, object]) -> str | None:
    """Return a reason string if the call must be gated under auto mode, else None."""
    if tool_name in GATED_TOOLS:
        return f'{tool_name} mutates under auto mode — manual approval required'
    if tool_name in ACTION_GATED_TOOLS:
        action = str(tool_input.get('action') or '').strip().lower()
        if action not in READONLY_ACTIONS:
            return f'{tool_name}(action={action or "<unspecified>"}) mutates under auto mode — manual approval required'
    return None


@boundary
def main() -> int:
    payload = PreToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    if payload.permission_mode != 'auto':
        return 0
    # tool_input is always a Mapping today (pydantic rejects non-Mapping before this);
    # the guard is forward-looking defense-in-depth — if JsonObject ever loosens, an
    # unexpected shape coerces to {} and gates (fail closed) rather than crashing.
    tool_input = payload.tool_input if isinstance(payload.tool_input, Mapping) else {}
    reason = _gate_reason(payload.tool_name, tool_input)
    if reason is None:
        return 0
    if (GATE_DIR / f'disabled-{payload.session_id}').exists():
        return 0
    output = PreToolUseHookOutput(
        hook_specific_output=PreToolUseSpecificOutput(
            permission_decision='ask',
            permission_decision_reason=reason,
        ),
    )
    print(output.model_dump_json())
    return 0


if __name__ == '__main__':
    sys.exit(main())
