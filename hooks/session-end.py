#!/usr/bin/env -S uv run --quiet --script
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

import pydantic
from local_lib.session_tracker import SessionManager
from local_lib.types import SessionEndReason
from local_lib.utils import Timer

# Start timing
timer = Timer()


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)


class SessionEndHookInput(BaseModel):
    """SessionEnd hook input schema"""

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: str
    reason: SessionEndReason


# Read and validate hook input from stdin
hook_data = SessionEndHookInput.model_validate_json(sys.stdin.read())

# Mark session as ended using SessionManager (atomic with file locking)
with SessionManager(hook_data.cwd) as manager:
    manager.end_session(hook_data.session_id)

# Print all session information (comprehensive - don't prematurely optimize)
print(f'Completed in {timer.elapsed_ms()} ms')
print(repr(hook_data))
