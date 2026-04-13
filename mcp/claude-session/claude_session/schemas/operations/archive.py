"""
Archive operation schemas.

Models for session archive creation and format detection.

Version 2.0 introduces explicit artifact models with structural metadata,
replacing the implicit filename-keyed mappings of v1.x.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal

import pydantic

from claude_session.schemas.base import StrictModel
from claude_session.schemas.session.models import SessionRecord, Task
from claude_session.schemas.types import Base64JsonBytes, JsonDatetime, ToolResultExtension

__all__ = [
    'ARCHIVE_FORMAT_VERSION',
    'AgentFileEntry',
    'ArchiveMetadata',
    'FileMetadata',
    'MainSessionFileEntry',
    'PlanFileEntry',
    'SessionArchive',
    'SessionEnvEntry',
    'TodoFileEntry',
    'ToolResultDirectoryEntry',
    'ToolResultDirectoryFileEntry',
    'ToolResultEntry',
    'parse_agent_metadata',
]


# -- Archive Format Version ----------------------------------------------------

ARCHIVE_FORMAT_VERSION = '2.2'
"""Current archive format version. Used when creating new archives.

Version history:
- 2.2: Added session_env, session_memory, and debug_log
- 2.1: Added tool_result_dirs for pdf-<uuid>/page-NN.jpg directory structures
- 2.0: Explicit artifact models, tasks support, agent structure preservation
"""


# -- Archive Metadata (MCP Tool Response) --------------------------------------


class FileMetadata(StrictModel):
    """Metadata about a single JSONL file in the archive."""

    filename: str
    record_count: int


class ArchiveMetadata(StrictModel):
    """
    Metadata about created archive.

    Returned by save_current_session MCP tool.
    """

    file_path: str
    session_id: str
    format: Literal['json', 'zst']
    size_mb: float  # Size in megabytes, rounded to 2 decimal places
    archived_at: datetime
    session_records: int  # Records in main {session_id}.jsonl file
    agent_records: int  # Records in agent-*.jsonl files
    file_count: int  # Number of JSONL files included (main + agents)
    files: Sequence[FileMetadata]  # Per-file breakdown
    custom_title: str | None = None  # User-defined session name from /rename (if any)


# -- Session Archive V2 - Explicit Artifact Models (Composite-First Ordering) ---
#
# Design principles:
# - Store only source-of-truth fields (filenames derivable from IDs)
# - Explicit structure (no implicit path parsing)
# - Metadata before data (record_count before records)
# - Top-down ordering: composite model first, then constituents


class SessionArchive(StrictModel):
    """
    Archive format v2.x - explicit artifact models.

    Version history:
    - 2.2: Added session_env, session_memory, and debug_log
    - 2.1: Added tool_result_dirs for directory-based tool results (pdf page renders)
    - 2.0: Explicit artifact models, tasks support, agent structure preservation

    Derived fields (not stored, computed on restore):
    - Main session filename: f"{session_id}.jsonl"
    - Agent filename: f"agent-{agent_id}.jsonl"
    - Agent relative path: "subagents/..." if nested else just filename
    """

    # Core metadata
    version: str  # "2.0"
    session_id: str
    archived_at: JsonDatetime
    original_project_path: str
    claude_code_version: str  # Required in v2 (use current or "unknown")
    machine_id: str | None = None
    custom_title: str | None = None

    # Explicit artifact collections
    main_session: MainSessionFileEntry
    agent_files: Sequence[AgentFileEntry] = ()
    plan_files: Sequence[PlanFileEntry] = ()
    tool_results: Sequence[ToolResultEntry] = ()
    tool_result_dirs: Sequence[ToolResultDirectoryEntry] = ()
    todos: Sequence[TodoFileEntry] = ()
    session_env: Sequence[SessionEnvEntry] = ()
    tasks: Sequence[Task] = ()  # Canonical Task model from session/models.py
    task_metadata: Mapping[str, str] = pydantic.Field(
        default_factory=dict
    )  # filename -> content (.highwatermark, etc.)
    session_env: Sequence[SessionEnvEntry]
    session_memory: str | None  # session-memory/summary.md content
    debug_log: str | None  # debug/<session-id>.txt content

    # Statistics (for quick inspection without iterating records)
    total_session_records: int
    total_agent_records: int


# -- V2 Entry Models (Constituents) --------------------------------------------


class MainSessionFileEntry(StrictModel):
    """Main session file content.

    Derived fields:
    - filename: f"{parent.session_id}.jsonl"
    - location: Always flat in project folder (never nested)
    """

    record_count: int  # Metadata first
    records: Sequence[SessionRecord]


class AgentFileEntry(StrictModel):
    """Agent file with structural metadata.

    Derived fields (computed on restore):
    - filename: f"agent-{agent_id}.jsonl"
    - relative_path: f"subagents/agent-{agent_id}.jsonl" if nested
                     else f"agent-{agent_id}.jsonl"

    The nested flag indicates directory structure:
    - nested=False: File in <project-folder>/agent-{id}.jsonl (pre-2.1.2)
    - nested=True: File in <project-folder>/<sid>/subagents/agent-{id}.jsonl (2.1.2+)

    Agent ID patterns (Claude Code 2.1.25+):
    - Plain hex: "5271c147" (agent_type=None)
    - Typed: "aprompt_suggestion-a12dbf" (agent_type="aprompt_suggestion")
    """

    agent_id: str  # Source of truth (e.g., "aprompt_suggestion-d7f1a0")
    agent_type: str | None = None  # Parsed type prefix, or None for plain hex
    nested: bool  # True if in subagents/ directory
    record_count: int  # Metadata first
    records: Sequence[SessionRecord]


class PlanFileEntry(StrictModel):
    """Plan file content.

    Derived fields:
    - filename: f"{slug}.md"
    - location: ~/.claude/plans/{slug}.md
    """

    slug: str
    content: str


class ToolResultEntry(StrictModel):
    """Tool result file content.

    Derived fields:
    - filename: f"{tool_use_id}{extension}"
    - location: projects/<enc>/<sid>/tool-results/{filename}
    """

    tool_use_id: str
    content: Base64JsonBytes
    extension: ToolResultExtension


class ToolResultDirectoryFileEntry(StrictModel):
    """A file within a tool result directory (e.g., page-01.jpg in pdf-<uuid>/).

    Derived fields:
    - location: projects/<enc>/<sid>/tool-results/{parent.name}/{filename}
    """

    filename: str
    content: Base64JsonBytes
    extension: ToolResultExtension


class ToolResultDirectoryEntry(StrictModel):
    """A tool result directory (e.g., pdf-<uuid>/ containing page images).

    Derived fields:
    - location: projects/<enc>/<sid>/tool-results/{name}/
    """

    name: str
    files: Sequence[ToolResultDirectoryFileEntry]


class SessionEnvEntry(StrictModel):
    """Session-env file content.

    Derived fields:
    - location: ~/.claude/session-env/{session_id}/{filename}

    Session-env files are small text files written by hooks.
    """

    filename: str
    content: str


class TodoFileEntry(StrictModel):
    """Todo file content.

    Derived fields:
    - filename: f"{session_id}-agent-{agent_id}.json"
    - location: ~/.claude/todos/{filename}

    Note: content is opaque JSON string - structure may vary by Claude Code version.
    """

    agent_id: str  # Which agent this todo belongs to
    content: str  # JSON string (not parsed, may vary)


def parse_agent_metadata(filename: str) -> tuple[str, str | None]:
    """
    Parse agent ID and type from filename.

    Examples:
        "agent-5271c147.jsonl" -> ("5271c147", None)
        "agent-aprompt_suggestion-a12dbf.jsonl" -> ("aprompt_suggestion-a12dbf", "aprompt_suggestion")

    Pattern detection:
        agent-<type>-<6-hex>.jsonl -> typed agent
        agent-<8-hex>.jsonl -> plain agent (no type)
        agent-<anything-else>.jsonl -> treat as plain agent ID

    Args:
        filename: Agent filename like "agent-xxx.jsonl"

    Returns:
        (agent_id, agent_type) tuple
    """
    # Remove "agent-" prefix and ".jsonl" suffix
    base = filename.removeprefix('agent-').removesuffix('.jsonl')

    # Check for typed agent pattern: <type>-<6-hex>
    if '-' in base:
        # Split from the RIGHT to handle types with dashes (e.g., "some-type-abc123")
        parts = base.rsplit('-', 1)
        if len(parts) == 2 and len(parts[1]) == 6:
            # Looks like type-hex pattern
            agent_type = parts[0]
            return (base, agent_type)

    # Plain hex or unknown pattern - no type
    return (base, None)
