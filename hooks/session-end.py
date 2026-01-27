#!/usr/bin/env -S uv run --quiet --script
"""SessionEnd hook for Claude Code session tracking.

See: https://code.claude.com/docs/en/hooks#sessionend
"""

# /// script
# dependencies = [
#   "pydantic>=2.0.0",
#   "local_lib",
# ]
#
# [tool.uv.sources]
# local_lib = { path = "../local-lib/", editable = true }
# ///
from __future__ import annotations

import sys
from pathlib import Path

from local_lib.schemas.hooks import SessionEndHookInput
from local_lib.session_tracker import SessionManager
from local_lib.utils import Timer

# Start timing
timer = Timer()

# Read and validate hook input from stdin
hook_data = SessionEndHookInput.model_validate_json(sys.stdin.read())

# Update session tracking (atomic with file locking)
with SessionManager(hook_data.cwd) as manager:
    if Path(hook_data.transcript_path).exists():
        manager.end_session(hook_data.session_id, reason=hook_data.reason)
        print(f'Completed in {timer.elapsed_ms()} ms')
        print(f'session_id: {hook_data.session_id}')
        print(repr(hook_data))
    else:
        manager.remove_empty_session(hook_data.session_id, hook_data.transcript_path)
        print(f'Session {hook_data.session_id} removed (no transcript) in {timer.elapsed_ms()} ms')
