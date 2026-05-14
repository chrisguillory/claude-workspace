"""Claude Code version extraction from session records (JSONL fallback).

Used by archive/info when sessions.json's metadata.claude_version is absent —
e.g., orphaned sessions on disk without a corresponding workspace entry.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from cc_lib.types import CCVersion
from packaging.version import InvalidVersion

from claude_session.introspection import get_cc_version_fields
from claude_session.schemas.session import SessionRecord

__all__ = [
    'get_version_from_records',
]

logger = logging.getLogger(__name__)


def get_version_from_records(records: Sequence[SessionRecord]) -> CCVersion | None:
    """Extract Claude Code version from session records.

    Walks records, finds the first one with a ``CCVersionMarker``-annotated
    field carrying a parseable PEP 440 string. Used when sessions.json doesn't
    have a workspace_version for the session.

    Returns CCVersion, or None if no parseable version found.
    """
    for record in records:
        for value in get_cc_version_fields(record).values():
            if not value:
                continue
            try:
                return CCVersion.parse(value)
            except InvalidVersion:
                logger.exception('Skipping unparseable record version: %r', value)
                continue
    return None
