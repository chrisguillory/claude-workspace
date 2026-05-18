"""JSON output envelopes for claude-remote-bash CLI commands."""

from __future__ import annotations

from collections.abc import Sequence

from cc_lib.schemas import ClosedModel

from claude_remote_bash.discovery import DiscoveredHost

__all__ = [
    'DiscoverResult',
    'GroupInfo',
]


class DiscoverResult(ClosedModel):
    """Top-level JSON output of ``claude-remote-bash discover --format json``."""

    remote_daemons: Sequence[DiscoveredHost]
    """Daemons advertised by other machines on the LAN; never includes the local machine."""

    local_daemon: DiscoveredHost | None = None
    """The local machine's own daemon if its mDNS advertisement is on the wire; ``None`` otherwise."""

    groups: Sequence[GroupInfo] = ()
    """Named host groups from ``client_config.json``."""


class GroupInfo(ClosedModel):
    """A named host group from ``client_config.json``."""

    name: str
    """Group name as it appears in ``client_config.json`` and on ``--target``."""

    members: Sequence[str]
    """Host aliases in declaration order. Stale entries stay listed."""
