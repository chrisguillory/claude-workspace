#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pydantic>=2.0.0",
# ]
# ///
"""Shared utilities for Claude Code session detection and management.

This module provides reusable functions for extracting session information
from Claude Code hook inputs, improving upon hardcoded path approaches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pydantic


class BaseModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)


class HookInput(BaseModel):
    """Base hook input schema for session-related hooks."""

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: str
    permission_mode: str


class SessionInfo(BaseModel):
    """Container for Claude Code session information."""

    session_id: str
    parent_id: str | None = None

    def __str__(self) -> str:
        if self.parent_id:
            return f'session_id: {self.session_id}\nparent_id: {self.parent_id}'
        return f'session_id: {self.session_id}'


def read_hook_input() -> HookInput:
    """Read and parse hook input from stdin.

    Returns:
        Validated HookInput model

    Raises:
        pydantic.ValidationError: If input doesn't match schema
    """
    return HookInput.model_validate_json(sys.stdin.read())


def get_session_info(hook_input: HookInput) -> SessionInfo:
    """Extract session information from Claude Code hook input.

    This function uses the transcript_path provided by Claude Code to
    read session metadata, avoiding hardcoded paths.

    Args:
        hook_input: HookInput model containing:
            - session_id: Current session UUID
            - transcript_path: Path to session transcript JSONL file

    Returns:
        SessionInfo object containing session_id and optionally parent_id

    Example:
        >>> hook_input = read_hook_input()
        >>> info = get_session_info(hook_input)
        >>> print(f"Session: {info.session_id}")
        >>> print(f"Parent: {info.parent_id}")
    """
    # Try to extract parent_id from transcript file
    parent_id = _extract_parent_id(hook_input)

    return SessionInfo(session_id=hook_input.session_id, parent_id=parent_id)


def _extract_parent_id(hook_input: HookInput) -> str | None:
    """Extract parent conversation ID (leafUuid) from transcript file.

    Args:
        hook_input: HookInput model with transcript_path

    Returns:
        Parent conversation UUID if found, None otherwise
    """
    transcript_file = Path(hook_input.transcript_path)

    # Handle case where transcript file doesn't exist yet
    if not transcript_file.exists():
        return None

    try:
        with open(transcript_file) as f:
            first_line = f.readline()

            # Handle empty file
            if not first_line:
                return None

            # Parse first line as JSON
            metadata: dict[str, str] = json.loads(first_line)

            # Extract leafUuid if present
            return metadata.get('leafUuid')

    except (OSError, json.JSONDecodeError):
        # Silently handle corrupted/unreadable files
        # (Hook should not fail the entire session)
        return None


def print_session_info(session_info: SessionInfo) -> None:
    """Print session information in a formatted way.

    Args:
        session_info: SessionInfo object to print
    """
    print(str(session_info))


# Example usage when run as a hook
if __name__ == '__main__':
    # This allows the module to be used directly as a hook
    try:
        hook_data = read_hook_input()
        info = get_session_info(hook_data)
        print_session_info(info)
    except Exception as e:
        # Hooks should not crash - print error and continue
        print(f'Error detecting session: {e}', file=sys.stderr)
        sys.exit(0)  # Exit successfully to not block Claude
