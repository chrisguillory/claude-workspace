#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
#     "document-search",
#     "pydantic>=2",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../../../cc-lib/", editable = true }
# document-search = { path = "../../../mcp/document-search/", editable = true }
# ///
"""Gather PR context + index the session for the create-pr skill.

Collects git/GitHub context and indexes the session transcript for semantic
search, then emits structured text for the skill prompt. Tailored for
chrisguillory/claude-workspace; drafts follow the repo's CLAUDE.md PR
guidelines.

Usage: gather-pr-context.py [BASE_BRANCH]   (BASE_BRANCH defaults to 'main')
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import SubsetModel
from document_search.schemas.indexing import IndexingResult


class ExistingPR(SubsetModel):
    """PR metadata from gh pr view, if one exists for this branch."""

    number: int
    title: str
    state: str
    body: str | None = None


class SessionInfo(SubsetModel):
    """Session metadata from claude-session info."""

    session_id: str
    session_file: str
    project_path: str


class PRContext(SubsetModel):
    """Top-level container for all gathered PR context."""

    branch: str
    base_branch: str
    existing_pr: ExistingPR | None
    commits: Sequence[str]
    files_changed: str
    scratch_files: Sequence[str]
    session_id: str | None
    session_indexed: bool
    session_chunks: int


@ErrorBoundary(exit_code=1)
def main() -> None:
    base = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else 'main'
    print(PRContextGatherer(base).run())


class PRContextGatherer:
    """Gather git, GitHub, and session context for PR creation."""

    def __init__(self, base_branch: str = 'main') -> None:
        self._base = base_branch

    def run(self) -> str:
        branch = self._git('rev-parse', '--abbrev-ref', 'HEAD')
        existing_pr = self._check_existing_pr()
        commits = self._git('log', f'{self._base}..HEAD', '--oneline').splitlines()
        files_changed = self._git('diff', f'{self._base}...HEAD', '--stat')
        scratch_files = self._find_scratch_files()
        session_id, session_chunks = self._index_session()

        ctx = PRContext(
            branch=branch,
            base_branch=self._base,
            existing_pr=existing_pr,
            commits=commits,
            files_changed=files_changed,
            scratch_files=scratch_files,
            session_id=session_id,
            session_indexed=session_chunks > 0,
            session_chunks=session_chunks,
        )
        return self._format_output(ctx)

    def _git(self, *args: str) -> str:
        result = subprocess.run(['git', *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def _check_existing_pr(self) -> ExistingPR | None:
        result = subprocess.run(
            ['gh', 'pr', 'view', '--json', 'number,title,state,body'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return ExistingPR.model_validate(json.loads(result.stdout))

    def _find_scratch_files(self) -> Sequence[str]:
        scratch_dir = Path.cwd() / 'scratch'
        if not scratch_dir.is_dir():
            return []
        return sorted(f.name for f in scratch_dir.glob('pr-*.md'))

    def _index_session(self) -> tuple[str | None, int]:
        """Index the session transcript + directory. Returns (session_id, total chunks).

        Degrades gracefully: a failed index (e.g. a transient embedding-provider
        overload) is surfaced on stderr and reported as 0 chunks, never silently.
        """
        result = subprocess.run(
            ['claude-session', 'info', '--format', 'json'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print('claude-session info failed — session indexing skipped', file=sys.stderr)
            return None, 0

        session = SessionInfo.model_validate(json.loads(result.stdout))
        transcript = Path(session.session_file)
        session_dir = transcript.parent / session.session_id

        paths = [str(transcript)]
        if session_dir.is_dir():
            paths.append(str(session_dir))

        print(f'Indexing session ({len(paths)} paths)...', file=sys.stderr)
        result = subprocess.run(
            ['document-search', 'index', *paths, '-c', 'document-chunks', '--no-gitignore', '--format', 'json'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f'Index failed (search the prior index instead): {result.stderr.strip()[:200]}', file=sys.stderr)
            return session.session_id, 0

        idx = IndexingResult.model_validate(json.loads(result.stdout))
        return session.session_id, idx.chunks_created + idx.chunks_skipped

    def _format_output(self, ctx: PRContext) -> str:
        lines: list[str] = []

        if ctx.existing_pr:
            lines.append(f'Mode: UPDATE (PR #{ctx.existing_pr.number}: {ctx.existing_pr.title})')
            lines.append(f'State: {ctx.existing_pr.state}')
        else:
            lines.append('Mode: CREATE (no existing PR for this branch)')
        lines.append('')

        lines.append(f'Branch: {ctx.branch}')
        lines.append(f'Base: {ctx.base_branch}')
        if ctx.session_id:
            lines.append(f'Session: {ctx.session_id}')
        lines.append('')

        lines.append(f'## Commits ({len(ctx.commits)})')
        lines.append('')
        lines.extend(f'  {c}' for c in ctx.commits[:30])
        if len(ctx.commits) > 30:
            lines.append(f'  ... and {len(ctx.commits) - 30} more')
        lines.append('')

        lines.append('## Files changed')
        lines.append('')
        lines.append(ctx.files_changed)
        lines.append('')

        lines.append('## Session')
        if ctx.session_indexed:
            lines.append(f'Indexed: {ctx.session_chunks} chunks (transcript + session directory)')
            lines.append(
                'Search via mcp__document-search__search_documents (collection "document-chunks") for decisions, feedback, research.'
            )
        else:
            lines.append(
                '(indexing skipped — search the existing index, or draft from git context + this conversation)'
            )
        lines.append('')

        if ctx.scratch_files:
            lines.append('## Existing scratch files')
            lines.append('')
            lines.extend(f'  - {f}' for f in ctx.scratch_files)

        return '\n'.join(lines)


if __name__ == '__main__':
    main()
