"""Pydantic schemas for Claude Code workspace."""

from __future__ import annotations

from cc_lib.schemas.base import CamelModel, ClosedModel, OpenModel, StrictModel, SubsetModel
from cc_lib.schemas.hooks import (
    BashToolInput,
    EditToolInput,
    PostToolUseHookInput,
    PostToolUseHookOutput,
    PostToolUseSpecificOutput,
    PreToolUseHookInput,
    PreToolUseHookOutput,
    PreToolUseSpecificOutput,
    SessionEndHookInput,
    SessionStartHookInput,
    SessionStartHookOutput,
    SessionStartSpecificOutput,
    SyncHookOutput,
    UserPromptSubmitHookOutput,
    UserPromptSubmitSpecificOutput,
    WriteToolInput,
)

__all__ = [
    'BashToolInput',
    'CamelModel',
    'ClosedModel',
    'EditToolInput',
    'OpenModel',
    'PostToolUseHookInput',
    'PostToolUseHookOutput',
    'PostToolUseSpecificOutput',
    'PreToolUseHookInput',
    'PreToolUseHookOutput',
    'PreToolUseSpecificOutput',
    'SessionEndHookInput',
    'SessionStartHookInput',
    'SessionStartHookOutput',
    'SessionStartSpecificOutput',
    'StrictModel',
    'SubsetModel',
    'SyncHookOutput',
    'UserPromptSubmitHookOutput',
    'UserPromptSubmitSpecificOutput',
    'WriteToolInput',
]
