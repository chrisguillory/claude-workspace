"""Wire-protocol messages exchanged between client and daemon over TCP."""

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
    'MountRequest',
    'MountResponse',
    'TunnelOk',
    'TunnelOpen',
    'UnmountRequest',
    'UnmountResponse',
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


# -- Filesystem mount ---------------------------------------------------------


class MountRequest(ClosedModel):
    """Client → Daemon: spawn crb-nfsd serving ``root``.

    Reuses an existing nfsd keyed by ``(root, readonly)`` if one is already
    running; otherwise spawns a fresh child and returns enough info for the
    client to set up a tunnel to it.

    ``root`` is interpreted on the daemon (peer) side. ``~`` expands to the
    daemon's user home; absolute paths are taken as-is. The daemon validates
    the resolved path against its export-allowlist policy before spawning.
    """

    type: Literal['mount'] = 'mount'
    id: str
    root: str
    readonly: bool = False


class MountResponse(ClosedModel):
    """Daemon → Client: nfsd is ready, here's how to claim a tunnel to it.

    The named ``(root, readonly)`` crb-nfsd is running and accepting NFS RPCs
    on the daemon's loopback. The client opens new TCP connections to the
    daemon, sends `TunnelOpen(mount_id=...)` on each, and those connections
    become byte-pipes to the nfsd's port.
    """

    type: Literal['mount_response'] = 'mount_response'
    id: str
    mount_id: str
    """Opaque identifier the client uses to claim tunnel connections."""


class TunnelOpen(ClosedModel):
    """Client → Daemon: claim this PSK-authed TCP connection as a tunnel.

    The daemon connects to the crb-nfsd loopback port registered under
    ``mount_id`` and bidirectionally pipes bytes. Once `TunnelOk` is written
    by the daemon, both sides treat the connection as raw bytes — not framed
    messages.
    """

    type: Literal['tunnel_open'] = 'tunnel_open'
    mount_id: str


class TunnelOk(ClosedModel):
    """Daemon → Client: tunnel established.

    Bytes after this message frame are raw NFS RPC traffic in both directions.
    """

    type: Literal['tunnel_ok'] = 'tunnel_ok'
    mount_id: str


class UnmountRequest(ClosedModel):
    """Client → Daemon: release the client's hold on ``mount_id``.

    When the refcount hits zero the daemon SIGTERMs the underlying crb-nfsd
    child. Always succeeds (idempotent).
    """

    type: Literal['unmount'] = 'unmount'
    id: str
    mount_id: str


class UnmountResponse(ClosedModel):
    """Daemon → Client: unmount acknowledged.

    ``child_terminated`` is True when this was the last holder of the mount
    and the daemon killed the nfsd child; False when other holders remain.
    """

    type: Literal['unmount_response'] = 'unmount_response'
    id: str
    mount_id: str
    child_terminated: bool


# -- Error --------------------------------------------------------------------


class ErrorResponse(ClosedModel):
    """Daemon → Client: error response for any request."""

    type: Literal['error'] = 'error'
    id: str | None = None
    message: str


# -- Discriminated union ------------------------------------------------------

Message = Annotated[
    AuthRequest
    | AuthOk
    | AuthFail
    | ExecuteRequest
    | ExecuteResult
    | MountRequest
    | MountResponse
    | TunnelOpen
    | TunnelOk
    | UnmountRequest
    | UnmountResponse
    | ErrorResponse,
    pydantic.Discriminator('type'),
]
