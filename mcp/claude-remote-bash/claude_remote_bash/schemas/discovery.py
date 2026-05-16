"""CLI output schemas for ``claude-remote-bash discover --format json``."""

from __future__ import annotations

from collections.abc import Sequence

from cc_lib.schemas.base import ClosedModel

__all__ = [
    'DiscoverResult',
    'DiscoveredHostInfo',
    'GroupInfo',
]


class DiscoveredHostInfo(ClosedModel):
    """A daemon discovered via mDNS — JSON serialization schema.

    Mirrors the fields of the internal ``discovery.DiscoveredHost`` class.
    """

    alias: str
    hostname: str
    ips: Sequence[str]
    port: int
    version: str
    is_self: bool


class GroupInfo(ClosedModel):
    """A named host group from ``client_config.json`` — JSON serialization schema.

    Members are alias strings in the order declared by the config. The
    discover command does not enrich them with daemon details; readers
    cross-reference ``DiscoverResult.daemons`` for that.
    """

    name: str
    members: Sequence[str]


class DiscoverResult(ClosedModel):
    """Top-level JSON output of ``claude-remote-bash discover --format json``.

    Wrapped in an object (rather than a bare array) so future fields
    (errors, cache state, scan duration) can be added without breaking
    consumers.
    """

    daemons: Sequence[DiscoveredHostInfo]
    groups: Sequence[GroupInfo] = ()
