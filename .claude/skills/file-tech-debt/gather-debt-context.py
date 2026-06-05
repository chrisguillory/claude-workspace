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
"""Gather tech-debt-filing context + index the session for the file-tech-debt skill.

The mirror of create-pr's gatherer, inverted target: a filed issue is a
lightweight pointer whose full depth lives in the indexed session transcript.
So this indexes the session (provenance), prints the session ID for the body
footer, and reports which scratch drafts + tech-debt/area/category labels
already exist so the skill reuses rather than duplicates.

Usage: gather-debt-context.py   (no arguments)
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

REPO = 'chrisguillory/claude-workspace'


class SessionInfo(SubsetModel):
    """Session metadata from claude-session info."""

    session_id: str
    session_file: str
    project_path: str


class DebtContext(SubsetModel):
    """Top-level container for all gathered tech-debt-filing context."""

    session_id: str | None
    session_indexed: bool
    session_chunks: int
    scratch_files: Sequence[str]
    labels: Sequence[str]


@ErrorBoundary(exit_code=1)
def main() -> None:
    print(DebtContextGatherer().run())


class DebtContextGatherer:
    """Gather session + label + scratch context for tech-debt issue filing."""

    def run(self) -> str:
        session_id, session_chunks = self._index_session()
        ctx = DebtContext(
            session_id=session_id,
            session_indexed=session_chunks > 0,
            session_chunks=session_chunks,
            scratch_files=self._find_scratch_files(),
            labels=self._list_labels(),
        )
        return self._format_output(ctx)

    def _find_scratch_files(self) -> Sequence[str]:
        scratch_dir = Path.cwd() / 'scratch'
        if not scratch_dir.is_dir():
            return []
        return sorted(f.name for f in scratch_dir.glob('issue-*.md'))

    def _list_labels(self) -> Sequence[str]:
        """Existing repo labels — so the skill knows which to create vs reuse."""
        result = subprocess.run(
            ['gh', 'label', 'list', '--repo', REPO, '--limit', '200', '--json', 'name'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print('gh label list failed — assume no labels exist yet', file=sys.stderr)
            return []
        return sorted(item['name'] for item in json.loads(result.stdout))

    def _index_session(self) -> tuple[str | None, int]:
        """Index the session transcript + directory. Returns (session_id, total chunks).

        Degrades gracefully: a failed index (e.g. a transient embedding-provider
        overload) is surfaced on stderr and reported as 0 chunks, never silently.
        The session ID is still returned for the provenance footer.
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

    def _format_output(self, ctx: DebtContext) -> str:
        lines: list[str] = []

        if ctx.session_id:
            lines.append(f'Session: {ctx.session_id}')
        lines.append('')

        lines.append('## Session')
        if ctx.session_indexed:
            lines.append(f'Indexed: {ctx.session_chunks} chunks (transcript + session directory)')
            lines.append(
                'Search via mcp__document-search__search_documents (collection "document-chunks") '
                'to recover the finding(s) and their full reasoning.'
            )
        else:
            lines.append('(indexing skipped — search the existing index, or file from this conversation)')
        lines.append('')

        lines.append('## Existing labels')
        lines.append('')
        debt = [n for n in ctx.labels if n == 'tech-debt' or n.startswith(('area:', 'category:', 'severity:'))]
        if debt:
            lines.extend(f'  - {n}' for n in debt)
        else:
            lines.append('  (none of tech-debt / area:* / category:* / severity:* exist — create what you use)')
        lines.append('')

        if ctx.scratch_files:
            lines.append('## Existing scratch drafts')
            lines.append('')
            lines.extend(f'  - {f}' for f in ctx.scratch_files)

        return '\n'.join(lines)


if __name__ == '__main__':
    main()
