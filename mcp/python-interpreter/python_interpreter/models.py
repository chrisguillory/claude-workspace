"""Pydantic models for the Python Interpreter MCP server."""

from __future__ import annotations

import datetime
import typing
from collections.abc import Mapping

import pydantic

__all__ = [
    'StrictModel',
    'InterpreterInfo',
    'InterpreterState',
    'InterpreterSource',
    'SavedInterpreterConfig',
    'InterpreterRegistry',
    'SessionInfo',
    'ExecuteRequest',
    'DriverExecuteResponse',
    'DriverListVarsResponse',
    'DriverResetResponse',
    'DriverReadyResponse',
]

type InterpreterState = typing.Literal['running', 'stopped']
type InterpreterSource = typing.Literal['builtin', 'saved', 'transient']


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation - no extra fields, all fields required unless Optional."""

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


class InterpreterInfo(StrictModel):
    """API response model for interpreter information."""

    # Identity
    name: str
    source: InterpreterSource
    state: InterpreterState

    # Configuration
    python_path: str | None
    cwd: str | None
    has_startup_script: bool
    description: str | None

    # Runtime (None when stopped)
    pid: int | None
    started_at: datetime.datetime | None
    uptime: str | None


class SavedInterpreterConfig(StrictModel, frozen=False):
    """Persisted interpreter configuration (no runtime state).

    Stored in {project_dir}/.claude/interpreters.json.
    python_path can be relative to project dir (e.g., '.venv/bin/python').
    """

    python_path: str
    cwd: str | None
    env: dict[str, str] | None
    startup_script: str | None
    description: str | None


class InterpreterRegistry(StrictModel, frozen=False):
    """Registry of saved interpreter configurations.

    Persisted to {project_dir}/.claude/interpreters.json.
    frozen=False allows mutation for add/remove operations.
    """

    discover_pycharm: bool
    interpreters: Mapping[str, SavedInterpreterConfig]


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
    interpreter: str


class DriverExecuteResponse(StrictModel):
    """Response from driver execute action."""

    stdout: str
    stderr: str
    result: str
    error: str | None
    error_type: str | None
    module_name: str | None


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
