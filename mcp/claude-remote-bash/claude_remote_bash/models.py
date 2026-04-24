"""Pydantic models for the claude-remote-bash protocol."""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic
from cc_lib.schemas.base import ClosedModel

__all__ = [
    'AuthFail',
    'AuthOk',
    'AuthRequest',
    'ErrorResponse',
    'ExecuteRequest',
    'ExecuteResult',
    'Message',
]


# -- Authentication -----------------------------------------------------------


class AuthRequest(ClosedModel):
    """Client → Daemon: authenticate with pre-shared key."""

    type: Literal['auth'] = 'auth'
    key: str


class AuthOk(ClosedModel):
    """Daemon → Client: authentication succeeded."""

    type: Literal['auth_ok'] = 'auth_ok'
    alias: str
    hostname: str
    os: str
    user: str
    shell: str
    version: str


class AuthFail(ClosedModel):
    """Daemon → Client: authentication failed."""

    type: Literal['auth_fail'] = 'auth_fail'
    reason: str


# -- Command execution --------------------------------------------------------


class ExecuteRequest(ClosedModel):
    """Client → Daemon: execute a command."""

    type: Literal['execute'] = 'execute'
    id: str
    command: str
    session_id: str
    agent_id: str | None = None
    timeout: float = 120.0


class ExecuteResult(ClosedModel):
    """Daemon → Client: command execution result.

    stdout and stderr are distinct fields so clients can route each to its
    matching local descriptor — tools that key error detection off stderr
    continue to work across the wire.
    """

    type: Literal['result'] = 'result'
    id: str
    stdout: str
    stderr: str
    exit_code: int
    cwd: str


# -- Error --------------------------------------------------------------------


class ErrorResponse(ClosedModel):
    """Daemon → Client: error response for any request."""

    type: Literal['error'] = 'error'
    id: str | None = None
    message: str


# -- Discriminated union ------------------------------------------------------

Message = Annotated[
    AuthRequest | AuthOk | AuthFail | ExecuteRequest | ExecuteResult | ErrorResponse,
    pydantic.Discriminator('type'),
]
