"""Shared exceptions for grok-kit.

Exception Hierarchy:
    GrokKitError (base)
    └── AuthExpiredError (grok.com returned 401/403)
"""

from __future__ import annotations

__all__ = [
    'AuthExpiredError',
    'GrokKitError',
]


class GrokKitError(Exception):
    """Base for all grok-kit-specific errors."""


class AuthExpiredError(GrokKitError):
    """grok.com returned 401/403; cookies invalid or expired.

    The MCP/CLI surface relays this to the user with the refresh instruction;
    callers should not attempt inline re-authentication.
    """

    REFRESH_INSTRUCTION = 'Run `grok-kit auth login` to refresh cookies.'

    def __init__(self, status_code: int) -> None:
        super().__init__(f'grok.com returned HTTP {status_code} — auth expired or invalid. {self.REFRESH_INSTRUCTION}')
        self.status_code = status_code
