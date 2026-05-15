"""mDNS browse-result cache persisted in ``hosts-cache.json``."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from cc_lib.schemas import ClosedModel
from cc_lib.utils.atomic_write import atomic_write

from claude_remote_bash.discovery import DiscoveredHost
from claude_remote_bash.paths import DATA_DIR, HOSTS_CACHE

__all__ = [
    'CACHE_TTL_SECONDS',
    'HostEntry',
    'HostsCache',
]

CACHE_TTL_SECONDS = 30.0
"""How long a browse result is reused before another mDNS sweep is required."""


class HostEntry(ClosedModel):
    """One daemon, as stored in the browse cache."""

    alias: str
    hostname: str
    ips: Sequence[str]
    port: int
    is_self: bool


class HostsCache(ClosedModel):
    """An mDNS browse result with the timestamp it was captured."""

    timestamp: float
    """Unix epoch seconds when the browse completed."""

    hosts: Sequence[HostEntry]

    @classmethod
    def load(cls, *, max_age_s: float | None = CACHE_TTL_SECONDS) -> HostsCache | None:
        """Return the cached snapshot, or ``None`` if missing or older than ``max_age_s``.

        Pass ``max_age_s=None`` to skip the age check entirely — useful for
        best-effort consumers (e.g. shell completion) where a stale alias list
        beats no completion.

        Missing file → ``None``. Malformed JSON or unexpected shape raises.
        """
        if not HOSTS_CACHE.exists():
            return None
        cache = cls.model_validate_json(HOSTS_CACHE.read_text())
        if max_age_s is not None and time.time() - cache.timestamp > max_age_s:
            return None
        return cache

    @classmethod
    def from_browse(cls, hosts: Sequence[DiscoveredHost]) -> HostsCache:
        """Snapshot a fresh mDNS browse result, stamping it with the current time."""
        entries = [
            HostEntry(alias=h.alias, hostname=h.hostname, ips=list(h.ips), port=h.port, is_self=h.is_self)
            for h in hosts
        ]
        return cls(timestamp=time.time(), hosts=entries)

    def write(self) -> None:
        """Persist this snapshot atomically.

        The explicit ``chmod 0o700`` is load-bearing: ``atomic_write``'s
        ``mkdir`` respects umask (typically 0o755). The cache file itself
        is set to 0o600 via ``atomic_write``'s ``mode=`` argument.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(DATA_DIR, 0o700)
        atomic_write(HOSTS_CACHE, self.model_dump_json().encode(), mode=0o600)
