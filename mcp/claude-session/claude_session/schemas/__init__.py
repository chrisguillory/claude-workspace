"""
Schema definitions for claude-session-mcp.

This package contains Pydantic models for various data schemas:
- session: Claude Code session JSONL record types
- operations: Service operation result schemas (archive, restore, delete, lineage)

Future additions:
- api: Claude API request/response schemas (v1/messages, etc.)
"""

from __future__ import annotations

from claude_session.schemas.base import StrictModel
from claude_session.schemas.types import JsonDatetime, ModelId

__all__ = [
    'JsonDatetime',
    'ModelId',
    'StrictModel',
]
