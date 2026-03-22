"""Pydantic schemas for Claude Code workspace."""

from __future__ import annotations

from cc_lib.schemas.base import CamelModel, ClosedModel, OpenModel, StrictModel
from cc_lib.schemas.hooks import (
    BashToolInput,
    EditToolInput,
    PostToolUseHookInput,
    PreToolUseDecision,
    PreToolUseHookInput,
    PreToolUseHookOutput,
    SessionEndHookInput,
    SessionStartHookInput,
    WriteToolInput,
)

__all__ = [
    'BashToolInput',
    'CamelModel',
    'ClosedModel',
    'EditToolInput',
    'OpenModel',
    'PostToolUseHookInput',
    'PreToolUseDecision',
    'PreToolUseHookInput',
    'PreToolUseHookOutput',
    'SessionEndHookInput',
    'SessionStartHookInput',
    'StrictModel',
    'WriteToolInput',
]
