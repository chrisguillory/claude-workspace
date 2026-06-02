"""Archive operation schemas.

Models for session archive creation and format detection.

Version 2.0 introduces explicit artifact models with structural metadata,
replacing the implicit filename-keyed mappings of v1.x.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

import pydantic

from claude_session.schemas.base import StrictModel
from claude_session.schemas.session.models import SessionRecord, Task
from claude_session.schemas.types import Base64JsonBytes, JsonDatetime, ToolResultExtension
from claude_session.types import ArchiveFormat

__all__ = [
    'ARCHIVE_FORMAT_VERSION',
    'AgentFileEntry',
    'AgentMetadataEntry',
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

ARCHIVE_FORMAT_VERSION = '2.4'
"""Current archive format version. Used when creating new archives.

Version history:
- 2.4: Workflow-nested layout — agent_metadata workflow_subpath, run journals, run metadata
- 2.3: Added agent_metadata for agent-*.meta.json sidecars (Claude Code 2.1.70+)
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
    """Metadata about created archive.

    Returned by save_current_session MCP tool.
    """

    file_path: str
    session_id: str
    format: ArchiveFormat
    size_mb: float  # Size in megabytes, rounded to 2 decimal places
    archived_at: datetime
    session_records: int  # Records in main {session_id}.jsonl file
    agent_records: int  # Records in agent-*.jsonl files
    file_count: int  # Number of JSONL files included (main + agents)
    files: Sequence[FileMetadata]  # Per-file breakdown
    custom_title: str | None = None  # User-defined session name from /rename (if any)


# -- Session Archive - Explicit Artifact Models (Composite-First Ordering) ------
#
# Design principles:
# - Store only source-of-truth fields (filenames derivable from IDs)
# - Explicit structure (no implicit path parsing)
# - Metadata before data (record_count before records)
# - Top-down ordering: composite model first, then constituents


class SessionArchive(StrictModel):
    """Archive format v2.x - explicit artifact models.

    Version history:
    - 2.4: Workflow-nested layout — agent_metadata workflow_subpath, run journals, run metadata
    - 2.3: Added agent_metadata for agent-*.meta.json sidecars (Claude Code 2.1.70+)
    - 2.2: Added session_env, session_memory, and debug_log
    - 2.1: Added tool_result_dirs for directory-based tool results (pdf page renders)
    - 2.0: Explicit artifact models, tasks support, agent structure preservation

    Derived fields (not stored, computed on restore):
    - Main session filename: f"{session_id}.jsonl"
    - Agent filename: f"agent-{agent_id}.jsonl"
    - Agent metadata filename: f"agent-{agent_id}.meta.json"
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
    agent_metadata: Sequence[AgentMetadataEntry] = ()
    workflow_journals: Mapping[str, str] = pydantic.Field(
        default_factory=dict
    )  # subagents-relative path -> verbatim journal.jsonl content
    workflow_runs: Mapping[str, str] = pydantic.Field(
        default_factory=dict
    )  # session-relative path -> verbatim workflows/ run metadata (.json) + script (.js)
    plan_files: Sequence[PlanFileEntry] = ()
    tool_results: Sequence[ToolResultEntry] = ()
    tool_result_dirs: Sequence[ToolResultDirectoryEntry] = ()
    todos: Sequence[TodoFileEntry] = ()
    session_env: Sequence[SessionEnvEntry] = ()
    tasks: Sequence[Task] = ()  # Canonical Task model from session/models.py
    task_metadata: Mapping[str, str] = pydantic.Field(
        default_factory=dict
    )  # filename -> content (.highwatermark, etc.)
    session_memory: str | None  # session-memory/summary.md content
    debug_log: str | None  # debug/<session-id>.txt content

    # Statistics (for quick inspection without iterating records)
    total_session_records: int
    total_agent_records: int


# -- Entry Models (Constituents) ------------------------------------------------


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
    - relative_path: <sid>/subagents/[<workflow_subpath>/]agent-{agent_id}.jsonl if nested,
                     else agent-{agent_id}.jsonl

    The nested flag indicates directory structure:
    - nested=False: File in <project-folder>/agent-{id}.jsonl (pre-2.1.2)
    - nested=True: File in <project-folder>/<sid>/subagents/agent-{id}.jsonl (2.1.2+)

    agent_id is opaque (Claude Code recognizes agent files by affix, not id shape);
    agent_type is the cosmetic fork-label slug of a typed id (a<slug>-<hex>), or None
    for an untyped a<hex> id.
    """

    agent_id: str  # Opaque agent id, e.g. "acompact-016ed80fcaa022ea"
    agent_type: str | None = None  # Cosmetic type slug (e.g. "compact"), or None
    nested: bool  # True if in subagents/ directory
    workflow_subpath: str | None = None  # "workflows/wf_<runId>" for workflow-nested agents, else None
    record_count: int  # Metadata first
    records: Sequence[SessionRecord]


class AgentMetadataEntry(StrictModel):
    """Per-subagent invocation manifest sidecar (Claude Code 2.1.70+).

    Claude Code writes a small JSON file when spawning each sidechain agent.
    The agent_id matches the paired AgentFileEntry when both writes landed.

    Derived fields (computed on restore):
    - filename: f"agent-{agent_id}.meta.json"
    - location: <project>/{sid}/subagents/[<workflow_subpath>/]agent-{agent_id}.meta.json
      (beside the paired transcript: directly nested, or under workflows/wf_<runId>/)

    Sparse by design: pre-2.1.70 sidechains have no sidecar, and write
    failures are silently swallowed. Content stored raw for forward
    compatibility with new fields.
    """

    agent_id: str  # Matches paired AgentFileEntry.agent_id
    workflow_subpath: str | None = None  # "workflows/wf_<runId>" for workflow agents, else None
    content: str  # Raw JSON of agent-{id}.meta.json


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


# Typed agent-id: a<slug>-<hex>, slug charset [a-z0-9_-] (fork-label and prompt-derived
# slugs), hex tail 6+. Untyped ids are a<hex> (no internal dash) and don't match.
TYPED_AGENT_ID = re.compile(r'^a([a-z0-9_-]+?)-([0-9a-f]{6,})$')


def parse_agent_metadata(filename: str) -> tuple[str, str | None]:
    """Decompose agent-<id>.jsonl into (opaque agentId, optional type slug).

    agentId is opaque per Claude Code (recognized via the agent-/.jsonl affixes). The
    type slug is cosmetic display metadata. The repo's own -clone-<prefix> provenance
    suffix is split off before type detection, so the type is read from the base id.

    Examples:
        "agent-a00608189fad1602b.jsonl"         -> ("a00608189fad1602b", None)
        "agent-acompact-016ed80fcaa022ea.jsonl" -> ("acompact-016ed80fcaa022ea", "compact")
    """
    agent_id = filename.removeprefix('agent-').removesuffix('.jsonl')
    base_id = agent_id.split('-clone-', 1)[0]
    match = TYPED_AGENT_ID.match(base_id)
    return (agent_id, match.group(1) if match else None)
