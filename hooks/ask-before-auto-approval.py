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
    # Editing
    'Edit',
    'Write',
    # Selenium profile-state (sensitive auth import)
    'mcp__selenium-browser__navigate_with_profile_state',
    # iMessage mutations (outgoing messages to external parties)
    'mcp__imessage-kit__send_message',
    # Google Workspace — Calendar mutations
    'mcp__google-workspace__create_calendar',
    'mcp__google-workspace__delete_calendar',
    'mcp__google-workspace__update_calendar',
    'mcp__google-workspace__share_calendar',
    'mcp__google-workspace__update_calendar_sharing',
    'mcp__google-workspace__remove_calendar_sharing',
    'mcp__google-workspace__create_event',  # legacy name, may or may not map
    'mcp__google-workspace__manage_event',  # actual tool (create/update/delete/rsvp)
    'mcp__google-workspace__move_event',
    'mcp__google-workspace__manage_focus_time',
    'mcp__google-workspace__manage_out_of_office',
    # Google Workspace — Gmail mutations
    'mcp__google-workspace__send_gmail_message',
    'mcp__google-workspace__draft_gmail_message',
    'mcp__google-workspace__manage_gmail_filter',
    'mcp__google-workspace__manage_gmail_label',
    'mcp__google-workspace__modify_gmail_message_labels',
    'mcp__google-workspace__batch_modify_gmail_message_labels',
    # Google Workspace — Drive mutations
    'mcp__google-workspace__create_drive_file',
    'mcp__google-workspace__create_drive_folder',
    'mcp__google-workspace__update_drive_file',
    'mcp__google-workspace__copy_drive_file',
    'mcp__google-workspace__manage_drive_access',
    'mcp__google-workspace__set_drive_file_permissions',
    # Google Workspace — Docs mutations
    'mcp__google-workspace__create_doc',
    'mcp__google-workspace__batch_update_doc',
    'mcp__google-workspace__modify_doc_text',
    'mcp__google-workspace__find_and_replace_doc',
    'mcp__google-workspace__insert_doc_elements',
    'mcp__google-workspace__insert_doc_image',
    'mcp__google-workspace__create_table_with_data',
    'mcp__google-workspace__update_paragraph_style',
    'mcp__google-workspace__update_doc_headers_footers',
    'mcp__google-workspace__import_to_google_doc',
    'mcp__google-workspace__export_doc_to_pdf',
    'mcp__google-workspace__manage_document_comment',
    # Google Workspace — Sheets mutations
    'mcp__google-workspace__create_spreadsheet',
    'mcp__google-workspace__create_sheet',
    'mcp__google-workspace__modify_sheet_values',
    'mcp__google-workspace__append_table_rows',
    'mcp__google-workspace__format_sheet_range',
    'mcp__google-workspace__resize_sheet_dimensions',
    'mcp__google-workspace__manage_conditional_formatting',
    'mcp__google-workspace__manage_spreadsheet_comment',
    # Google Workspace — Slides mutations
    'mcp__google-workspace__create_presentation',
    'mcp__google-workspace__batch_update_presentation',
    'mcp__google-workspace__manage_presentation_comment',
    # Google Workspace — Forms mutations
    'mcp__google-workspace__create_form',
    'mcp__google-workspace__batch_update_form',
    'mcp__google-workspace__set_publish_settings',
    # Google Workspace — Contacts mutations
    'mcp__google-workspace__manage_contact',
    'mcp__google-workspace__manage_contact_group',
    'mcp__google-workspace__manage_contacts_batch',
    # Google Workspace — Tasks mutations
    'mcp__google-workspace__manage_task',
    'mcp__google-workspace__manage_task_list',
    # Google Workspace — Chat mutations
    'mcp__google-workspace__send_message',
    'mcp__google-workspace__create_reaction',
    # Google Workspace — Apps Script mutations
    'mcp__google-workspace__create_script_project',
    'mcp__google-workspace__delete_script_project',
    'mcp__google-workspace__update_script_content',
    'mcp__google-workspace__manage_deployment',
    'mcp__google-workspace__create_version',
    'mcp__google-workspace__run_script_function',
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
