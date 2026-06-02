"""Workflow run artifacts: the <session>/workflows/ tree (sibling to subagents/).

Claude Code's Workflow tool writes, alongside the subagents/workflows/<runId>/ transcripts,
a <session>/workflows/ tree holding per-run metadata (wf_<runId>.json) and the run's script
source (scripts/<name>-<runId>.js). Run metadata is reference-dense — it embeds the agentIds
of the run's sidechain transcripts, the sessionId, and an absolute scriptPath — so it is
remapped on fork. The script is author-written source and is carried verbatim.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from claude_session.services.artifacts.agent_ids import apply_agent_id_mapping

__all__ = [
    'WORKFLOW_RUNS_DIRNAME',
    'collect_workflow_runs',
    'remap_workflow_run_json',
    'write_workflow_runs',
]


WORKFLOW_RUNS_DIRNAME = 'workflows'  # <session>/workflows/ — sibling to subagents/


def collect_workflow_runs(project_folder: Path, session_id: str) -> Mapping[str, str]:
    """Collect the <session>/workflows/ tree, keyed by path relative to the session dir.

    Keys look like "workflows/wf_<runId>.json" and "workflows/scripts/<name>-<runId>.js".
    Content is read verbatim. Empty when the directory is absent (no workflow runs).
    """
    runs_dir = project_folder / session_id / WORKFLOW_RUNS_DIRNAME
    if not runs_dir.exists():
        return {}
    session_dir = project_folder / session_id
    return {
        p.relative_to(session_dir).as_posix(): p.read_text(encoding='utf-8')
        for p in sorted(runs_dir.rglob('*'))
        if p.is_file()
    }


def write_workflow_runs(project_folder: Path, session_id: str, runs: Mapping[str, str]) -> int:
    """Write a {session-relative path -> content} map under <project>/<session_id>/.

    Keys carry the workflows/ prefix; parent directories are created as needed.
    """
    if not runs:
        return 0
    session_dir = project_folder / session_id
    count = 0
    for relpath, content in runs.items():
        path = session_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        count += 1
    return count


def remap_workflow_run_json(
    content: str,
    agent_id_mapping: Mapping[str, str],
    old_session_id: str,
    new_session_id: str,
    path_replacements: Mapping[str, str] | None = None,
) -> str:
    """Remap a run-metadata JSON's embedded references for a forked session.

    Remaps the agentIds (which point at sibling transcripts) and the sessionId (which also
    appears inside scriptPath). path_replacements applies any old->new project-path
    substitutions (raw and encoded forms) for cross-project forks. Only the .json run metadata
    passes through here; the .js script is author source and is carried verbatim.
    """
    result = apply_agent_id_mapping(content, agent_id_mapping).replace(old_session_id, new_session_id)
    for old, new in (path_replacements or {}).items():
        result = result.replace(old, new)
    return result
