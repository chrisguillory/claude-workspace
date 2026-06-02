from __future__ import annotations

__all__ = [
    'GranolaAuthError',
    'GranolaError',
    'GranolaUnsupportedClientError',
]


class GranolaError(Exception):
    """Base for every granola-kit error."""


class GranolaAuthError(GranolaError):
    """Auth could not produce a usable token — missing or undecryptable store, refused Keychain, or failed refresh."""


class GranolaUnsupportedClientError(GranolaError):
    """A 200 response carried an 'Unsupported client' envelope — Granola rejected the client identity/token."""
