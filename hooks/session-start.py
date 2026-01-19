#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pydantic>=2.0.0",
#   "packaging",
#   "local_lib",
# ]
#
# [tool.uv.sources]
# local_lib = { path = "../local-lib/", editable = true }
# ///
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pydantic
import packaging.version

from local_lib.session_tracker import SessionManager
from local_lib.types import SessionSource
from local_lib.utils import Timer
import psutil

# Start timing
timer = Timer()


def find_claude_pid() -> int | None:
    """Find Claude process PID by walking up the process tree."""
    current = os.getppid()

    for _ in range(20):  # Depth limit
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid=,comm='],
            capture_output=True,
            text=True,
        )

        if not result.stdout.strip():
            break

        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ''

        # Check if this is Claude
        if 'claude' in comm.lower():
            return current

        if ppid == 0:
            break

        current = ppid

    return None


def get_claude_version(claude_pid: int) -> str:
    """Extract Claude Code version from the running process's executable path.

    Uses psutil to get the actual executable path of the Claude process,
    which contains the version (e.g., ~/.local/share/claude/versions/2.1.12).

    Args:
        claude_pid: PID of the running Claude process

    Returns:
        Version string (e.g., "2.1.12")

    Raises:
        packaging.version.InvalidVersion: If path doesn't contain valid version
        psutil.NoSuchProcess: If the process no longer exists
    """
    exe_path = Path(psutil.Process(claude_pid).exe())
    version = packaging.version.Version(exe_path.name)
    return str(version)


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)


class SessionStartHookInput(BaseModel):
    """SessionStart hook input schema"""

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: str
    source: SessionSource
    model: str | None = None  # Added in Claude Code v2.1.9 (only on startup, not resume)


# Read and validate hook input from stdin
hook_data = SessionStartHookInput.model_validate_json(sys.stdin.read())

# Use cwd as project directory (source of truth)
project_dir = hook_data.cwd

# Verify: encode cwd and check it matches transcript_path encoding
transcript_path = Path(hook_data.transcript_path)
encoded_project = transcript_path.parent.name
encoded_from_cwd = project_dir.replace('/', '-')
if not encoded_from_cwd.startswith('-'):
    encoded_from_cwd = '-' + encoded_from_cwd

encoding_matches = encoded_project == encoded_from_cwd

# Find Claude PID
claude_pid = find_claude_pid()

if claude_pid is None:
    print(f'Error: Could not find Claude PID. Hook data: {hook_data}', file=sys.stderr)
    sys.exit(1)

# Extract parent_id from transcript file
parent_id: str | None = None
transcript_file = Path(hook_data.transcript_path)

if transcript_file.exists():
    with open(transcript_file) as f:
        first_line = f.readline()
        if first_line:
            metadata = json.loads(first_line)
            if 'leafUuid' in metadata:
                parent_id = metadata['leafUuid']

# Get Claude version from running process
claude_version = get_claude_version(claude_pid)

# Track session using SessionManager (atomic with file locking)
with SessionManager(project_dir) as manager:
    crashed_ids = manager.detect_crashed_sessions()
    pruned_ids = manager.prune_orphaned_sessions()
    manager.start_session(
        session_id=hook_data.session_id,
        transcript_path=hook_data.transcript_path,
        source=hook_data.source,
        claude_pid=claude_pid,
        parent_id=parent_id,
        startup_model=hook_data.model,
        claude_version=claude_version,
    )

# Print session information
print(f'Completed in {timer.elapsed_ms()} ms')
print(repr(hook_data))
print(f'claude_pid: {claude_pid}')
print(f'claude_version: {claude_version}')
print(f'parent_id: {parent_id}')
print(f'encoding_verified: {encoding_matches}')
if crashed_ids:
    print(f'crashed_sessions: {crashed_ids}')
if pruned_ids:
    print(f'pruned_orphaned_sessions: {pruned_ids}')
