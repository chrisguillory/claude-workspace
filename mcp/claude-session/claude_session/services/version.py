"""Claude Code version extraction from session records (JSONL fallback).

Used by archive/info when sessions.json's metadata.claude_version is absent —
e.g., orphaned sessions on disk without a corresponding workspace entry.
"""

from __future__ import annotations

from collections.abc import Sequence

from cc_lib.types import CCVersion

from claude_session.schemas.session import (
    ApiErrorSystemRecord,
    AssistantRecord,
    CompactBoundarySystemRecord,
    InformationalSystemRecord,
    LocalCommandSystemRecord,
    SessionRecord,
    TurnDurationSystemRecord,
    UserRecord,
)

__all__ = [
    'VERSION_RECORD_TYPES',
    'get_version_from_records',
]


# All record types that have a version field
VERSION_RECORD_TYPES = (
    UserRecord,
    AssistantRecord,
    LocalCommandSystemRecord,
    CompactBoundarySystemRecord,
    ApiErrorSystemRecord,
    InformationalSystemRecord,
    TurnDurationSystemRecord,
)


def get_version_from_records(records: Sequence[SessionRecord]) -> CCVersion | None:
    """Extract Claude Code version from session records.

    Searches through records to find the first one with a version field.
    Used when sessions.json doesn't have a workspace_version for the session.

    Returns CCVersion, or None if no version found.
    """
    for record in records:
        if isinstance(record, VERSION_RECORD_TYPES) and record.version:
            return CCVersion(record.version)
    return None
