#!/usr/bin/env -S uv run --quiet --script
"""SessionEnd hook for Claude Code session tracking.

See: https://code.claude.com/docs/en/hooks#sessionend
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import sys
from pathlib import Path

from cc_lib.claude_context import find_claude_process
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.exceptions import RivalSessionError
from cc_lib.schemas.hooks import SessionEndHookInput
from cc_lib.session_tracker import SessionManager
from cc_lib.utils import Timer

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> None:
    timer = Timer()
    hook_data = SessionEndHookInput.model_validate_json(sys.stdin.read())
    claude_pid = find_claude_process().pid

    with SessionManager(hook_data.cwd) as manager:
        if Path(hook_data.transcript_path).exists():
            rival = manager.find_rival_session(hook_data.session_id, claude_pid)
            if rival is not None:
                raise RivalSessionError(
                    session_id=hook_data.session_id,
                    rival_pid=rival.metadata.claude_pid,
                    claude_pid=claude_pid,
                )
            manager.end_session(hook_data.session_id, reason=hook_data.reason)
            print(f'Completed in {timer.elapsed_ms()} ms')
            print(f'session_id: {hook_data.session_id}')
            print(repr(hook_data))
        else:
            manager.remove_empty_session(hook_data.session_id, hook_data.transcript_path)
            print(f'Session {hook_data.session_id} removed (no transcript) in {timer.elapsed_ms()} ms')


@boundary.handler(RivalSessionError)
def _handle_rival_session(exc: RivalSessionError) -> None:
    print(f'session-end: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
