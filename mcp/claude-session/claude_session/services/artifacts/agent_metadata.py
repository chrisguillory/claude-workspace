from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from claude_session.services.artifacts.agent_ids import AGENT_FILENAME_PATTERN

__all__ = [
    'AGENT_METADATA_FILENAME_PATTERN',
    'SubagentsDirectoryContents',
    'classify_subagents_directory',
    'collect_agent_metadata',
    'transform_agent_metadata_filename',
    'write_agent_metadata',
]


AGENT_METADATA_FILENAME_PATTERN = re.compile(
    r'^agent-'
    r'('
    r'(?:[a-z_]+-)?'  # Optional type prefix (typed agents, 2.1.25+)
    r'[a-f0-9]+'  # Required hex ID
    r'(?:-clone-[a-f0-9]+)?'  # Optional clone suffix
    r')'
    r'\.meta\.json$'
)


@dataclass(frozen=True)
class SubagentsDirectoryContents:
    """Single-pass classification of entries in a session's subagents/ directory.

    Callers route by category:
    - jsonl_files: agent-*.jsonl transcripts (filename match)
    - metadata_files: agent-*.meta.json sidecars (filename match)
    - subdirectories: unusual; cleanup candidate
    - unexpected_files: filename matched no known pattern (fail-fast)
    """

    jsonl_files: Sequence[Path]
    metadata_files: Sequence[Path]
    subdirectories: Sequence[Path]
    unexpected_files: Sequence[Path]


def classify_subagents_directory(
    project_folder: Path,
    session_id: str,
) -> SubagentsDirectoryContents | None:
    """Classify entries in <project_folder>/<session_id>/subagents/.

    Returns None if the directory doesn't exist (pre-2.1.2 flat layouts).
    """
    subagents_dir = project_folder / session_id / 'subagents'
    if not subagents_dir.exists():
        return None

    jsonl_files: list[Path] = []
    metadata_files: list[Path] = []
    subdirectories: list[Path] = []
    unexpected_files: list[Path] = []

    for path in subagents_dir.iterdir():
        if path.is_dir():
            subdirectories.append(path)
            continue
        if not path.is_file():
            unexpected_files.append(path)
            continue
        if AGENT_FILENAME_PATTERN.match(path.name):
            jsonl_files.append(path)
            continue
        if AGENT_METADATA_FILENAME_PATTERN.match(path.name):
            metadata_files.append(path)
            continue
        unexpected_files.append(path)

    return SubagentsDirectoryContents(
        jsonl_files=jsonl_files,
        metadata_files=metadata_files,
        subdirectories=subdirectories,
        unexpected_files=unexpected_files,
    )


def collect_agent_metadata(
    project_folder: Path,
    session_id: str,
) -> Mapping[str, str]:
    """Collect agent metadata sidecars for a session.

    Walks <project_folder>/<session_id>/subagents/ for agent-*.meta.json files.

    Args:
        project_folder: Project folder containing the session
        session_id: Session ID being collected

    Returns:
        Mapping of filename -> raw JSON content. Empty if no sidecars.
    """
    contents = classify_subagents_directory(project_folder, session_id)
    if contents is None:
        return {}
    return {p.name: p.read_text(encoding='utf-8') for p in contents.metadata_files}


def transform_agent_metadata_filename(
    old_filename: str,
    agent_id_mapping: Mapping[str, str],
) -> str:
    """Transform a meta.json filename using an agent ID mapping.

    Pattern: agent-{old_id}.meta.json -> agent-{new_id}.meta.json

    Args:
        old_filename: Original filename (e.g., "agent-5271c147.meta.json")
        agent_id_mapping: Mapping of old_agent_id -> new_agent_id

    Returns:
        Transformed filename

    Raises:
        ValueError: If filename doesn't match agent-*.meta.json pattern
        KeyError: If agent ID not found in the mapping
    """
    match = AGENT_METADATA_FILENAME_PATTERN.match(old_filename)
    if not match:
        raise ValueError(f"Filename doesn't match agent-*.meta.json pattern: {old_filename}")

    old_agent_id = match.group(1)
    if old_agent_id not in agent_id_mapping:
        raise KeyError(f'Agent ID not found in mapping: {old_agent_id}')

    new_agent_id = agent_id_mapping[old_agent_id]
    return f'agent-{new_agent_id}.meta.json'


def write_agent_metadata(
    project_folder: Path,
    session_id: str,
    metadata: Mapping[str, str],
) -> int:
    """Write agent metadata sidecars to <project>/<session_id>/subagents/.

    Caller ensures filenames already reflect any agent_id mapping
    (use transform_agent_metadata_filename for clone operations).

    Args:
        project_folder: Project folder for the target session
        session_id: Target session ID
        metadata: Mapping of filename -> raw JSON content

    Returns:
        Count of files written
    """
    if not metadata:
        return 0

    subagents_dir = project_folder / session_id / 'subagents'
    subagents_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for filename, content in metadata.items():
        path = subagents_dir / filename
        path.write_text(content, encoding='utf-8')
        count += 1
    return count
