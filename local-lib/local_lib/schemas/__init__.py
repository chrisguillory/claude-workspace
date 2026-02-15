"""Pydantic schemas for Claude Code workspace."""

from __future__ import annotations

from local_lib.schemas.hooks import (
    PreToolUseDecision,
    PreToolUseHookInput,
    PreToolUseHookOutput,
    SessionEndHookInput,
    SessionStartHookInput,
    StrictModel,
)

__all__ = [
    'PreToolUseDecision',
    'PreToolUseHookInput',
    'PreToolUseHookOutput',
    'SessionEndHookInput',
    'SessionStartHookInput',
    'StrictModel',
]
