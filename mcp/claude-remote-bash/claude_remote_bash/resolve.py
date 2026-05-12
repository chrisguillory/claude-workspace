"""Host resolution: browse mDNS, cache the result, look up an atom against it."""

from __future__ import annotations

from collections.abc import Sequence

from claude_remote_bash.cache import HostsCache
from claude_remote_bash.discovery import browse_hosts
from claude_remote_bash.exceptions import HostNotFoundError

__all__ = [
    'browse_and_cache',
    'browse_fresh_and_cache',
    'lookup_alias',
    'raise_host_not_found',
]


async def browse_and_cache() -> HostsCache:
    """Return the current view of discovered daemons — cache hit if fresh, else a fresh browse."""
    cached = HostsCache.load()
    if cached is not None:
        return cached
    return await browse_fresh_and_cache()


async def browse_fresh_and_cache() -> HostsCache:
    """Browse mDNS, persist the snapshot atomically, return it."""
    hosts = await browse_hosts(timeout=3.0)
    snapshot = HostsCache.from_browse(hosts)
    snapshot.write()
    return snapshot


def lookup_alias(cache: HostsCache, atom: str) -> tuple[Sequence[str], int] | None:
    """Resolve one atom against ``cache``. Pure — no I/O.

    The atom is either a literal ``ip:port`` (returns the parts directly)
    or a host alias matched case-insensitively. Returns ``None`` when the
    alias isn't in the cache.
    """
    if ':' in atom:
        parts = atom.rsplit(':', 1)
        return [parts[0]], int(parts[1])

    atom_lower = atom.lower()
    for entry in cache.hosts:
        if entry.alias.lower() == atom_lower:
            return list(entry.ips), entry.port

    return None


def raise_host_not_found(atom: str) -> None:
    """Raise ``HostNotFoundError`` with an actionable message naming the missing atom."""
    raise HostNotFoundError(
        f'Host not found: {atom}\n'
        'Run `claude-remote-bash discover` to see available hosts.\n'
        '\n'
        'Common causes:\n'
        "  - The daemon isn't running on the target.\n"
        '  - Alias typo — check the output of `discover`.\n'
        "  - Target is on a different LAN segment (mDNS doesn't cross subnets)."
    )
