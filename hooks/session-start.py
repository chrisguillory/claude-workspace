#!/usr/bin/env -S uv run --quiet --script
"""SessionStart hook for Claude Code session tracking.

See: https://code.claude.com/docs/en/hooks#sessionstart
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "packaging",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import packaging.version
import psutil
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.phantom import PhantomHandler
from cc_lib.schemas.base import SubsetModel
from cc_lib.schemas.hooks import SessionStartHookInput
from cc_lib.session_tracker import SessionManager, find_claude_pid
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

    claude_pid = find_claude_pid()
    claude_version = _get_claude_version(claude_pid)
    process_created_at = _get_process_created_at(claude_pid)

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


def _get_claude_version(claude_pid: int) -> str:
    """Extract Claude Code version from the running process's executable path."""
    exe_path = Path(psutil.Process(claude_pid).exe())
    return str(packaging.version.Version(exe_path.name))


def _get_process_created_at(claude_pid: int) -> datetime:
    """Get process creation time from the OS via psutil."""
    create_time = psutil.Process(claude_pid).create_time()
    return datetime.fromtimestamp(create_time, UTC).astimezone()


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


if __name__ == '__main__':
    main()
