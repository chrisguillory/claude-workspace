from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from claude_session.services.artifacts.agent_ids import AGENT_FILENAME_PATTERN

__all__ = [
    'AGENT_METADATA_FILENAME_PATTERN',
    'SubagentsDirectoryContents',
    'agent_metadata_relpath',
    'classify_subagents_directory',
    'collect_agent_metadata',
    'collect_workflow_journals',
    'transform_agent_metadata_filename',
    'transform_agent_metadata_relpath',
    'write_agent_metadata',
    'write_workflow_journals',
]


# Agent metadata sidecar: agent-<id>.meta.json, paired 1:1 with agent-<id>.jsonl. Same
# opaque id as AGENT_FILENAME_PATTERN (Claude Code derives the name by swapping the suffix).
AGENT_METADATA_FILENAME_PATTERN = re.compile(r'^agent-(.+)\.meta\.json$')


@dataclass(frozen=True)
class SubagentsDirectoryContents:
    """Single-pass classification of files under a session's subagents/ tree (recursive).

    Routing is by basename, so depth is irrelevant: workflow agents under
    subagents/workflows/wf_<runId>/ classify the same as top-level ones.

    - jsonl_files: agent-*.jsonl transcripts
    - metadata_files: agent-*.meta.json sidecars
    - journal_files: workflow run-journals (journal.jsonl)
    - unexpected_files: matched no known pattern (surface, never silently drop)
    """

    jsonl_files: Sequence[Path]
    metadata_files: Sequence[Path]
    journal_files: Sequence[Path]
    unexpected_files: Sequence[Path]


def classify_subagents_directory(
    project_folder: Path,
    session_id: str,
) -> SubagentsDirectoryContents | None:
    """Classify files under <project_folder>/<session_id>/subagents/ recursively.

    Returns None if the directory doesn't exist (pre-2.1.2 flat layouts).
    """
    subagents_dir = project_folder / session_id / 'subagents'
    if not subagents_dir.exists():
        return None

    jsonl_files: list[Path] = []
    metadata_files: list[Path] = []
    journal_files: list[Path] = []
    unexpected_files: list[Path] = []

    for path in sorted(subagents_dir.rglob('*')):
        if not path.is_file():
            continue
        if AGENT_FILENAME_PATTERN.match(path.name):
            jsonl_files.append(path)
        elif AGENT_METADATA_FILENAME_PATTERN.match(path.name):
            metadata_files.append(path)
        elif path.name == 'journal.jsonl':
            journal_files.append(path)
        else:
            unexpected_files.append(path)

    return SubagentsDirectoryContents(
        jsonl_files=jsonl_files,
        metadata_files=metadata_files,
        journal_files=journal_files,
        unexpected_files=unexpected_files,
    )


def collect_agent_metadata(
    project_folder: Path,
    session_id: str,
) -> Mapping[str, str]:
    """Collect agent metadata sidecars for a session, keyed by path relative to subagents/.

    Keys preserve workflow nesting (e.g. "workflows/wf_<runId>/agent-<id>.meta.json"), so a
    write restores each sidecar beside its paired transcript. Top-level sidecars key on the
    bare filename.

    Args:
        project_folder: Project folder containing the session
        session_id: Session ID being collected

    Returns:
        Mapping of subagents-relative POSIX path -> raw JSON content. Empty if no sidecars.
    """
    contents = classify_subagents_directory(project_folder, session_id)
    if contents is None:
        return {}
    return _collect_relpath_contents(project_folder / session_id / 'subagents', contents.metadata_files)


def collect_workflow_journals(
    project_folder: Path,
    session_id: str,
) -> Mapping[str, str]:
    """Collect Workflow run-journals for a session, keyed by path relative to subagents/.

    Keys carry the workflow nesting (e.g. "workflows/wf_<runId>/journal.jsonl"). Content is
    carried verbatim; callers apply agent-id/session-id remap on fork.

    Returns:
        Mapping of subagents-relative POSIX path -> raw journal content. Empty if none.
    """
    contents = classify_subagents_directory(project_folder, session_id)
    if contents is None:
        return {}
    return _collect_relpath_contents(project_folder / session_id / 'subagents', contents.journal_files)


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


def transform_agent_metadata_relpath(
    relpath: str,
    agent_id_mapping: Mapping[str, str],
) -> str:
    """Remap the agent-id in a sidecar's subagents-relative path, preserving its subdir.

    "agent-{old}.meta.json"                      -> "agent-{new}.meta.json"
    "workflows/wf_<runId>/agent-{old}.meta.json" -> "workflows/wf_<runId>/agent-{new}.meta.json"
    """
    parent, sep, base = relpath.rpartition('/')
    return f'{parent}{sep}{transform_agent_metadata_filename(base, agent_id_mapping)}'


def agent_metadata_relpath(agent_id: str, workflow_subpath: str | None) -> str:
    """subagents-relative path for an agent's sidecar, preserving any workflow nesting.

    None                   -> "agent-{agent_id}.meta.json"
    "workflows/wf_<runId>" -> "workflows/wf_<runId>/agent-{agent_id}.meta.json"
    """
    filename = f'agent-{agent_id}.meta.json'
    return f'{workflow_subpath}/{filename}' if workflow_subpath else filename


def write_agent_metadata(
    project_folder: Path,
    session_id: str,
    metadata: Mapping[str, str],
) -> int:
    """Write agent metadata sidecars under <project>/<session_id>/subagents/.

    Keys are POSIX paths relative to subagents/ and may carry a workflows/wf_<runId>/
    prefix; parent directories are created as needed. Callers apply any agent_id remap
    to the keys first (transform_agent_metadata_relpath for clone/restore).

    Args:
        project_folder: Project folder for the target session
        session_id: Target session ID
        metadata: Mapping of subagents-relative path -> raw JSON content

    Returns:
        Count of files written
    """
    return _write_subagents_tree(project_folder, session_id, metadata)


def write_workflow_journals(
    project_folder: Path,
    session_id: str,
    journals: Mapping[str, str],
) -> int:
    """Write Workflow run-journals under <project>/<session_id>/subagents/.

    Keys are subagents-relative POSIX paths (workflows/wf_<runId>/journal.jsonl); parent
    directories are created as needed. Callers apply any agent_id/session_id remap first.

    Returns:
        Count of files written
    """
    return _write_subagents_tree(project_folder, session_id, journals)


def _collect_relpath_contents(subagents_dir: Path, paths: Sequence[Path]) -> Mapping[str, str]:
    """Read files into a {subagents-relative POSIX path -> content} map."""
    return {p.relative_to(subagents_dir).as_posix(): p.read_text(encoding='utf-8') for p in paths}


def _write_subagents_tree(project_folder: Path, session_id: str, files: Mapping[str, str]) -> int:
    """Write a {subagents-relative path -> content} map under <project>/<session_id>/subagents/."""
    if not files:
        return 0
    subagents_dir = project_folder / session_id / 'subagents'
    count = 0
    for relpath, content in files.items():
        path = subagents_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        count += 1
    return count
