from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import pydantic
from cc_lib.schemas import ClosedModel
from cc_lib.utils.atomic_write import atomic_write
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel

__all__ = [
    'CACHE_TTL_SECONDS',
    'DeviceCache',
    'cache_path',
    'read_devices',
    'write_devices',
]

CACHE_TTL_SECONDS = 60.0
"""Cache freshness window — bounds how stale TAB completion can be."""


class DeviceCache(ClosedModel):
    """A snapshot of one host's Core Audio inputs + outputs.

    JSON output uses camelCase aliases per CLAUDE.md (``fetchedAt``). Old cache
    files written with snake_case keys still parse via ``validate_by_name=True``
    inherited from ClosedModel — smooth transition, no manual cache eviction needed.
    """

    model_config = ConfigDict(alias_generator=to_camel)

    hub: str
    fetched_at: float
    outputs: Sequence[str]
    inputs: Sequence[str]


def cache_path(hub: str) -> Path:
    """Return the per-hub cache file path under ``$XDG_CACHE_HOME/claude-remote-audio/devices/``."""
    base = os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return Path(base) / 'claude-remote-audio' / 'devices' / f'{hub}.json'


def read_devices(hub: str) -> DeviceCache | None:
    """Return cached devices for ``hub`` if present, parseable, and fresh — else ``None``.

    Any unusable cache (missing, TTL-expired, corrupt JSON, schema-drifted,
    unreadable) returns ``None`` so completion treats it as a miss and triggers
    a fresh dispatch. Apply overwrites the hub's cache on every run.
    """
    path = cache_path(hub)
    if not path.exists():
        return None
    try:
        cache = DeviceCache.model_validate_json(path.read_text())
    except (pydantic.ValidationError, json.JSONDecodeError, OSError):
        return None
    if time.time() - cache.fetched_at > CACHE_TTL_SECONDS:
        return None
    return cache


def write_devices(hub: str, *, outputs: Sequence[str], inputs: Sequence[str]) -> DeviceCache:
    """Persist a fresh snapshot for ``hub`` atomically; return the written cache."""
    cache = DeviceCache(hub=hub, fetched_at=time.time(), outputs=list(outputs), inputs=list(inputs))
    atomic_write(cache_path(hub), cache.model_dump_json().encode(), mode=0o600)
    return cache
