"""Shared utilities for Claude Code hooks, MCP servers, and scripts."""

from __future__ import annotations

from cc_lib.background_tasks import BackgroundTaskGroup
from cc_lib.batch_loader import GenericBatchLoader
from cc_lib.cli import add_install_command, create_app, run_app
from cc_lib.concurrency_tracker import ConcurrencyTracker
from cc_lib.error_boundary import ErrorBoundary, ErrorHandler
from cc_lib.library_boundary import LibraryBoundary
from cc_lib.utils.atomic_write import atomic_write

__all__ = [
    'BackgroundTaskGroup',
    'ConcurrencyTracker',
    'ErrorBoundary',
    'ErrorHandler',
    'GenericBatchLoader',
    'LibraryBoundary',
    'add_install_command',
    'atomic_write',
    'create_app',
    'run_app',
]
