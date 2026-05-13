"""Move operation schemas.

Models for session move results.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from claude_session.schemas.base import StrictModel

__all__ = [
    'MoveResult',
]


class MoveResult(StrictModel):
    """Result of a session move operation."""

    session_id: str
    source_project: str
    target_project: str

    # Aggregate counts
    files_moved: int
    files_deleted: int

    # Per-artifact-type breakdown (target-side moves)
    main_session_moved: int
    agent_files_moved: int
    agent_metadata_moved: int
    tool_results_moved: int
    session_memory_moved: bool

    paths_translated: bool

    was_running: bool
    was_terminated: bool
    backup_path: str | None
    custom_title: str | None

    resume_command: str
    was_dry_run: bool
    duration_ms: float
    moved_at: datetime
    warnings: Sequence[str]
