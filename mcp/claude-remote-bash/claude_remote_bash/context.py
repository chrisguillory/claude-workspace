"""Session context store for CWD tracking across stateless commands."""

from __future__ import annotations

import time
from collections.abc import Mapping

__all__ = [
    'SessionContext',
    'SessionContextStore',
]

# Default TTL: 24 hours
DEFAULT_TTL_SECONDS = 86400


class SessionContext:
    """Tracked state for one (session_id, agent_id) pair on this daemon."""

    def __init__(self, *, cwd: str, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        self.cwd = cwd
        self.command_count = 0
        self.ttl = ttl
        self._last_active = time.monotonic()

    def touch(self) -> None:
        """Update last-active timestamp."""
        self._last_active = time.monotonic()

    def is_expired(self) -> bool:
        """True if the context has been idle longer than its TTL."""
        return (time.monotonic() - self._last_active) > self.ttl

    def update_cwd(self, cwd: str) -> None:
        """Update the tracked working directory and bump activity."""
        self.cwd = cwd
        self.command_count += 1
        self.touch()


class SessionContextStore:
    """In-memory store of session contexts, keyed by (session_id, agent_id).

    Contexts auto-create on first access and expire after a configurable TTL.
    Expired entries are cleaned up lazily on access.
    """

    def __init__(self, *, default_cwd: str, ttl: float = DEFAULT_TTL_SECONDS) -> None:
        self._contexts: dict[tuple[str, str | None], SessionContext] = {}
        self._default_cwd = default_cwd
        self._ttl = ttl

    def get(self, session_id: str, agent_id: str | None = None) -> SessionContext:
        """Get or create a session context. Expired contexts are replaced."""
        key = (session_id, agent_id)
        ctx = self._contexts.get(key)

        if ctx is not None and ctx.is_expired():
            del self._contexts[key]
            ctx = None

        if ctx is None:
            ctx = SessionContext(cwd=self._default_cwd, ttl=self._ttl)
            self._contexts[key] = ctx

        return ctx

    def remove(self, session_id: str, agent_id: str | None = None) -> bool:
        """Remove a session context. Returns True if it existed."""
        return self._contexts.pop((session_id, agent_id), None) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired contexts. Returns the number removed."""
        expired = [k for k, v in self._contexts.items() if v.is_expired()]
        for k in expired:
            del self._contexts[k]
        return len(expired)

    def stats(self) -> Mapping[str, int]:
        """Return store statistics."""
        active = sum(1 for v in self._contexts.values() if not v.is_expired())
        return {
            'total': len(self._contexts),
            'active': active,
            'expired': len(self._contexts) - active,
        }
