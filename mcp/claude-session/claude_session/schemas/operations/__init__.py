"""
Operation schemas for service results.

This package contains Pydantic models for operation results returned by services.
These are extracted from service files to enable reuse and cleaner separation.
"""

from __future__ import annotations

from claude_session.schemas.operations.archive import (
    ARCHIVE_FORMAT_VERSION,
    ArchiveMetadata,
    FileMetadata,
    SessionArchive,
)
from claude_session.schemas.operations.context import SessionContext, SessionSource, SessionState
from claude_session.schemas.operations.delete import ArtifactFile, DeleteManifest, DeleteResult
from claude_session.schemas.operations.discovery import SessionInfo
from claude_session.schemas.operations.gist import GistArchiveResult
from claude_session.schemas.operations.lineage import (
    LineageEntry,
    LineageFile,
    LineageTree,
    LineageTreeNode,
)
from claude_session.schemas.operations.move import MoveResult
from claude_session.schemas.operations.restore import RestoreResult

__all__ = [
    # Archive
    'ARCHIVE_FORMAT_VERSION',
    'FileMetadata',
    'ArchiveMetadata',
    'SessionArchive',
    # Context
    'SessionContext',
    'SessionSource',
    'SessionState',
    # Delete
    'ArtifactFile',
    'DeleteManifest',
    'DeleteResult',
    # Discovery
    'SessionInfo',
    # Gist
    'GistArchiveResult',
    # Lineage
    'LineageEntry',
    'LineageFile',
    'LineageTree',
    'LineageTreeNode',
    # Move
    'MoveResult',
    # Restore
    'RestoreResult',
]
