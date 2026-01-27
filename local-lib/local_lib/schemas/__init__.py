"""Pydantic schemas for Claude Code workspace."""

from __future__ import annotations

from local_lib.schemas.hooks import (
    SessionEndHookInput,
    SessionStartHookInput,
    StrictModel,
)

__all__ = [
    'SessionEndHookInput',
    'SessionStartHookInput',
    'StrictModel',
]
