"""Pydantic models for the Python Interpreter MCP server."""

from __future__ import annotations

import datetime
import typing

import pydantic

__all__ = [
    'StrictModel',
    'InterpreterInfo',
    'SessionInfo',
    'ExecuteRequest',
    'DriverExecuteResponse',
    'DriverListVarsResponse',
    'DriverResetResponse',
    'DriverReadyResponse',
]

type InterpreterType = typing.Literal['builtin', 'external']


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


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
    interpreter: str = 'builtin'


class DriverExecuteResponse(StrictModel):
    """Response from driver execute action."""

    stdout: str
    stderr: str
    result: str
    error: str | None
    error_type: str | None = None
    module_name: str | None = None


class DriverListVarsResponse(StrictModel):
    """Response from driver list_vars action."""

    result: str
    error: str | None


class DriverResetResponse(StrictModel):
    """Response from driver reset action."""

    result: str
    error: str | None


class DriverReadyResponse(StrictModel):
    """Response from driver on startup (ready signal)."""

    status: typing.Literal['ready']
    python_version: str
    python_executable: str
