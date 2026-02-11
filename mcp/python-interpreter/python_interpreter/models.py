"""Pydantic models for the Python Interpreter MCP server."""

from __future__ import annotations

import datetime
import typing

import pydantic

__all__ = [
    'StrictModel',
    'TruncationInfo',
    'InterpreterInfo',
    'SessionInfo',
    'ExecuteRequest',
]

type InterpreterType = typing.Literal['builtin', 'external']


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


class TruncationInfo(StrictModel):
    """Information about truncated output."""

    file_path: str
    original_size: int
    truncated_at: int


class InterpreterInfo(StrictModel):
    """API response model for interpreter information."""

    name: str
    type: InterpreterType
    python_path: str | None = None
    cwd: str | None = None
    pid: int | None = None
    started_at: datetime.datetime | None = None
    uptime: str | None = None
    has_startup_script: bool = False


class SessionInfo(StrictModel):
    """Session and server metadata."""

    session_id: str
    project_dir: str
    socket_path: str
    transcript_path: str
    output_dir: str
    claude_pid: int
    started_at: datetime.datetime
    uptime: str


class ExecuteRequest(StrictModel):
    """Request body for HTTP execute endpoint."""

    code: str
    interpreter: str | None = None
