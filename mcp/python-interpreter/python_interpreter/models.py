"""Pydantic models for the Python Interpreter MCP server."""

from __future__ import annotations

import datetime
import typing
from collections.abc import Mapping

from cc_lib.schemas.base import ClosedModel

__all__ = [
    'DriverExecuteResponse',
    'DriverReadyResponse',
    'ExecuteRequest',
    'InterpreterInfo',
    'InterpreterRegistry',
    'InterpreterSource',
    'InterpreterState',
    'JetBrainsRunConfig',
    'JetBrainsSDKEntry',
    'SavedInterpreterConfig',
]

type InterpreterState = typing.Literal['running', 'stopped']
type InterpreterSource = typing.Literal['builtin', 'saved', 'jetbrains-sdk', 'jetbrains-run']


class InterpreterInfo(ClosedModel):
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


class SavedInterpreterConfig(ClosedModel):
    """Persisted interpreter configuration (no runtime state).

    Stored in ~/.claude-workspace/python_interpreter/interpreters.json.
    python_path can be relative to project dir (e.g., '.venv/bin/python').
    """

    python_path: str
    cwd: str | None
    env: Mapping[str, str] | None
    startup_script: str | None
    description: str | None


class InterpreterRegistry(ClosedModel):
    """Registry of saved interpreter configurations.

    Persisted to ~/.claude-workspace/python_interpreter/interpreters.json.
    """

    discover_jetbrains: bool
    interpreters: Mapping[str, SavedInterpreterConfig]


class JetBrainsSDKEntry(ClosedModel):
    """Python interpreter from JetBrains jdk.table.xml."""

    name: str
    python_path: str
    version: str | None
    flavor: str | None
    associated_project: str | None


class JetBrainsRunConfig(ClosedModel):
    """JetBrains 'Run with Python Console' configuration from .run.xml."""

    name: str
    xml_path: str
    python_path: str | None
    cwd: str | None
    env: Mapping[str, str] | None
    script_name: str | None
    parameters: str | None


class ExecuteRequest(ClosedModel):
    """Request body for HTTP execute endpoint."""

    code: str
    interpreter: str


class DriverExecuteResponse(ClosedModel):
    """Response from driver execute action."""

    stdout: str
    stderr: str
    result: str
    error: str | None
    error_type: str | None
    module_name: str | None


class DriverReadyResponse(ClosedModel):
    """Response from driver on startup (ready signal)."""

    status: typing.Literal['ready']
    python_version: str
    python_executable: str
