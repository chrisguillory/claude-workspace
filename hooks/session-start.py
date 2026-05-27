#!/usr/bin/env -S uv run --quiet --script
"""SessionStart hook for Claude Code session tracking.

See: https://code.claude.com/docs/en/hooks#sessionstart
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "click",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from cc_lib import os_process
from cc_lib.claude_context import find_claude_process
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.exceptions import RivalSessionError
from cc_lib.phantom import PhantomHandler
from cc_lib.schemas.base import SubsetModel
from cc_lib.schemas.hooks import SessionStartHookInput
from cc_lib.session_tracker import SessionManager
from cc_lib.types import CCVersion
from cc_lib.utils import Timer

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> None:
    timer = Timer()
    hook_data = SessionStartHookInput.model_validate_json(sys.stdin.read())
    project_dir = hook_data.cwd

    # Verify path encoding matches transcript location
    transcript_path = Path(hook_data.transcript_path)
    encoded_project = transcript_path.parent.name
    encoded_from_cwd = project_dir.replace('/', '-')
    if not encoded_from_cwd.startswith('-'):
        encoded_from_cwd = '-' + encoded_from_cwd
    encoding_matches = encoded_project == encoded_from_cwd

    process = find_claude_process()
    claude_pid = process.pid
    claude_version = CCVersion(process.exe_path().name)
    process_created_at = process.created_at.astimezone()

    # Extract parent_id from first line of transcript
    parent_id = _extract_parent_id(Path(hook_data.transcript_path))

    # Track session
    with SessionManager(project_dir) as manager:
        manager.detect_crashed_sessions()
        phantom = PhantomHandler(manager, claude_pid, claude_version)

        if phantom.is_phantom(hook_data.session_id, hook_data.source, Path(hook_data.transcript_path)):
            manager.prune_orphaned_sessions()
        else:
            phantom.cleanup(hook_data.session_id)
            manager.prune_orphaned_sessions()
            rival = manager.find_rival_session(hook_data.session_id, claude_pid)
            if rival is not None:
                raise RivalSessionError(
                    session_id=hook_data.session_id,
                    rival_pid=rival.metadata.claude_pid,
                    claude_pid=claude_pid,
                )
            manager.start_session(
                session_id=hook_data.session_id,
                transcript_path=hook_data.transcript_path,
                source=hook_data.source,
                claude_pid=claude_pid,
                parent_id=parent_id,
                startup_model=hook_data.model,
                claude_version=claude_version,
                process_created_at=process_created_at,
            )

    phantom.log()

    print(f'Completed in {timer.elapsed_ms()} ms')
    print(repr(hook_data))
    print(f'claude_pid: {claude_pid}')
    print(f'claude_version: {claude_version}')
    print(f'process_created_at: {process_created_at}')
    print(f'parent_id: {parent_id}')
    print(f'encoding_verified: {encoding_matches}')


# -- Helpers ------------------------------------------------------------------


class _TranscriptFirstLine(SubsetModel):
    """First line of a transcript JSONL — only the field we need."""

    leafUuid: str | None = None


def _extract_parent_id(transcript_file: Path) -> str | None:
    """Extract parent_id (leafUuid) from first line of transcript file."""
    if not transcript_file.exists():
        return None
    with open(transcript_file) as f:
        first_line = f.readline()
        if not first_line:
            return None
        return _TranscriptFirstLine.model_validate_json(first_line).leafUuid


def _emit_to_tty(*lines: str) -> None:
    """Write to /dev/tty in red bold (bypasses Claude Code's hook stderr capture)."""
    with open('/dev/tty', 'w') as tty:
        for line in lines:
            click.secho(line, fg='red', bold=True, file=tty)


def _notify_macos(exc: RivalSessionError) -> None:
    """Show a desktop notification for cross-tab visibility (macOS only)."""
    body = (
        f'Refused duplicate resume — session {exc.session_id[:8]} owned by pid {exc.rival_pid}. '
        f'Terminated rival pid {exc.claude_pid}.'
    )
    subprocess.run(['osascript', '-e', f'display notification "{body}" with title "Claude Code"'], check=False)


@boundary.handler(RivalSessionError)
def _handle_rival_session(exc: RivalSessionError) -> None:
    msg_owner = f'session-start: {exc}'
    msg_kill = f'session-start: terminating rival claude pid {exc.claude_pid}'
    print(msg_owner, file=sys.stderr)
    print(msg_kill, file=sys.stderr)
    _emit_to_tty(msg_owner, msg_kill)
    _notify_macos(exc)
    os_process.terminate(exc.claude_pid)


if __name__ == '__main__':
    main()
