"""Exception hierarchy for claude-remote-bash.

All raisable errors are declared here. Each exception's ``__str__`` returns a
complete user-facing message so handlers can be formatting-free — they just
print the exception.

Hierarchy
---------

    Exception
    └── RemoteBashError          # library root (all our errors subclass this)
        ├── ProtocolError        # wire-format errors (raised by daemon + CLI)
        ├── AuthError            # no PSK or authentication rejected
        ├── HostNotFoundError    # mDNS didn't find the alias
        ├── HostUnreachableError # all advertised IPs failed to connect
        └── DaemonError          # daemon returned an ErrorResponse

Subclasses set the ``prefix`` class attribute to prepend a label (e.g. "Daemon
error: ...") via ``__str__``; classes with no prefix print the bare message.
Handlers dispatch on ``RemoteBashError`` for all expected errors and on
``Exception`` for anything unexpected (which adds the class name for
diagnostic clarity).
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    'AuthError',
    'DaemonError',
    'HostNotFoundError',
    'HostUnreachableError',
    'ProtocolError',
    'RemoteBashError',
]


class RemoteBashError(Exception):
    """Root exception for claude-remote-bash.

    Every exception raised by this package subclasses ``RemoteBashError``.
    Subclasses set the ``prefix`` class attribute to prepend a label to the
    message; ``__str__`` composes ``"{prefix}: {message}"`` when a prefix is
    set and returns the bare message otherwise.
    """

    prefix: ClassVar[str] = ''

    def __str__(self) -> str:
        msg = super().__str__()
        return f'{self.prefix}: {msg}' if self.prefix else msg


class ProtocolError(RemoteBashError):
    """Wire-format error: framing, size limits, malformed JSON."""

    prefix = 'Protocol error'


class AuthError(RemoteBashError):
    """Authentication failed or not configured."""


class HostNotFoundError(RemoteBashError):
    """Host alias could not be resolved via mDNS."""


class HostUnreachableError(RemoteBashError):
    """TCP connection to all advertised addresses of a host failed."""


class DaemonError(RemoteBashError):
    """Daemon returned an error response."""

    prefix = 'Daemon error'
