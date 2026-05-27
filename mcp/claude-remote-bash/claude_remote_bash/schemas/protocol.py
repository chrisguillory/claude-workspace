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
    timeout: float | None = None
    """Seconds before SIGKILL; ``None`` (default) runs without limit."""


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
    """Daemon → Client: nfsd is ready; use ``mount_id`` to open tunnel channels.

    The named ``(root, readonly)`` crb-nfsd is running and accepting NFS RPCs
    on the daemon's loopback. The client claims tunnel channels on its
    PSK-authed connection by sending ``TunnelOpen(mount_id=...)``; the daemon
    allocates a ``channel_id`` and returns it in ``TunnelOk``. Subsequent
    frames on that channel carry raw NFS RPC bytes in both directions.
    """

    type: Literal['mount_response'] = 'mount_response'
    id: str
    mount_id: str
    """Opaque identifier the client uses to claim tunnel connections."""


class TunnelOpen(ClosedModel):
    """Client → Daemon: allocate a tunnel channel for ``mount_id``.

    Sent on the control channel of an already-PSK-authed connection. The
    daemon connects to the crb-nfsd loopback port registered under
    ``mount_id``, allocates a non-zero ``channel_id``, and returns it via
    ``TunnelOk``. Subsequent frames on the allocated channel carry raw NFS
    RPC bytes; control-channel frames continue to flow alongside.
    """

    type: Literal['tunnel_open'] = 'tunnel_open'
    mount_id: str


class TunnelOk(ClosedModel):
    """Daemon → Client: tunnel established on the assigned channel.

    The daemon allocates a non-zero ``channel_id`` for the tunnel and routes
    subsequent frames on that channel to the underlying ``crb-nfsd`` loopback
    socket. The client sends tunneled NFS RPCs by writing frames on the
    same channel; control messages continue to flow on channel 0.
    """

    type: Literal['tunnel_ok'] = 'tunnel_ok'
    mount_id: str
    channel_id: int


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
