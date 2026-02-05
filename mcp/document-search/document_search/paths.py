"""Centralized file paths for document search.

All persistent file locations in one place for consistency.
Dashboard and MCP server share these paths for coordination.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    'COLLECTIONS_LOCK_PATH',
    'COLLECTIONS_STATE_PATH',
    'DASHBOARD_LOCK_PATH',
    'DASHBOARD_STATE_PATH',
    'DEBUG_LOG_PATH',
    'DOCUMENT_SEARCH_DIR',
    'INDEX_STATE_DIR',
    'WORKSPACE_DIR',
    'index_state_path',
]

# Base directories
WORKSPACE_DIR = Path.home() / '.claude-workspace'
DOCUMENT_SEARCH_DIR = WORKSPACE_DIR / 'document_search'

# Collection registry
COLLECTIONS_STATE_PATH = DOCUMENT_SEARCH_DIR / 'collections.json'
COLLECTIONS_LOCK_PATH = DOCUMENT_SEARCH_DIR / 'collections.lock'

# Dashboard coordination
DASHBOARD_STATE_PATH = DOCUMENT_SEARCH_DIR / 'dashboard.json'
DASHBOARD_LOCK_PATH = DOCUMENT_SEARCH_DIR / 'dashboard.lock'

# Debug logging (enable detailed server logs for troubleshooting)
DEBUG_LOG_PATH = DOCUMENT_SEARCH_DIR / 'server.log'

# Per-collection index state
INDEX_STATE_DIR = DOCUMENT_SEARCH_DIR / 'index_state'


def index_state_path(collection_name: str) -> Path:
    """Get index state file path for a specific collection.

    Raises:
        ValueError: If collection_name contains path traversal sequences.
    """
    path = INDEX_STATE_DIR / f'{collection_name}.json'
    if not path.resolve().is_relative_to(INDEX_STATE_DIR.resolve()):
        raise ValueError(f'Invalid collection name: {collection_name}')
    return path
