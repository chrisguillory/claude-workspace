"""Pydantic schemas for claude-remote-bash."""

from __future__ import annotations

from claude_remote_bash.schemas.discovery import (
    DiscoveredHostInfo,
    DiscoverResult,
)
from claude_remote_bash.schemas.protocol import (
    AuthFail,
    AuthOk,
    AuthRequest,
    ErrorResponse,
    ExecuteRequest,
    ExecuteResult,
    Message,
)

__all__ = [  # noqa: RUF022 — grouped by schema module, not alphabetical
    # Protocol — wire messages
    'AuthRequest',
    'AuthOk',
    'AuthFail',
    'ExecuteRequest',
    'ExecuteResult',
    'ErrorResponse',
    'Message',
    # Discovery — CLI JSON output
    'DiscoveredHostInfo',
    'DiscoverResult',
]
