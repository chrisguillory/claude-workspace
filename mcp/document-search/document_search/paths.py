"""Centralized file paths for document search.

All persistent file locations in one place for consistency.
Dashboard and MCP server share these paths for coordination.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    'DASHBOARD_LOCK_PATH',
    'DASHBOARD_STATE_PATH',
    'DOCUMENT_SEARCH_DIR',
    'WORKSPACE_DIR',
]

# Base directories
WORKSPACE_DIR = Path.home() / '.claude-workspace'
DOCUMENT_SEARCH_DIR = WORKSPACE_DIR / 'document_search'

# Dashboard coordination
DASHBOARD_STATE_PATH = DOCUMENT_SEARCH_DIR / 'dashboard.json'
DASHBOARD_LOCK_PATH = DOCUMENT_SEARCH_DIR / 'dashboard.lock'
