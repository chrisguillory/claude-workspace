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
"""Gather docket-filing context + index the session for the add-to-docket skill.

A docket entry is a lightweight pointer whose full depth lives in the indexed
session transcript. So this indexes the session (provenance), prints the session
ID for the entry footer, and reports the next NN + existing entries per docket
type so the skill numbers correctly and the human can steer grouping.

Usage: gather-context.py   (no arguments)
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import SubsetModel
from document_search.schemas.indexing import IndexingResult

DOCKET_TYPES = ('tech-debt', 'feature', 'follow-up', 'idea')


class SessionInfo(SubsetModel):
    """Session metadata from claude-session info."""

    session_id: str
    session_file: str
    project_path: str


class DocketContext(SubsetModel):
    """Top-level container for all gathered docket-filing context."""

    session_id: str | None
    session_indexed: bool
    session_chunks: int
    next_ids: Mapping[str, str]
    existing: Mapping[str, Sequence[str]]


@ErrorBoundary(exit_code=1)
def main() -> None:
    print(DocketContextGatherer().run())


class DocketContextGatherer:
    """Gather session provenance + per-type next-NN for docket filing."""

    def run(self) -> str:
        session_id, session_chunks = self._index_session()
        next_ids, existing = self._scan_docket()
        ctx = DocketContext(
            session_id=session_id,
            session_indexed=session_chunks > 0,
            session_chunks=session_chunks,
            next_ids=next_ids,
            existing=existing,
        )
        return self._format_output(ctx)

    def _scan_docket(self) -> tuple[Mapping[str, str], Mapping[str, Sequence[str]]]:
        """Per docket type: next sequential NN (max existing + 1, 2-digit) + existing entries.

        Numbering is per-directory; collisions (two PRs claiming the same NN) are caught by
        tests/docket/test_docket_invariants.py at PR time, not prevented here.
        """
        docket = Path.cwd() / 'docket'
        next_ids: dict[str, str] = {}
        existing: dict[str, Sequence[str]] = {}
        for type_name in DOCKET_TYPES:
            type_dir = docket / type_name
            entries = sorted(f.name for f in type_dir.glob('*-*.md')) if type_dir.is_dir() else []
            existing[type_name] = entries
            max_n = max((int(e.split('-', 1)[0]) for e in entries if e.split('-', 1)[0].isdigit()), default=0)
            next_ids[type_name] = f'{max_n + 1:02d}'
        return next_ids, existing

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

    def _format_output(self, ctx: DocketContext) -> str:
        lines: list[str] = []

        if ctx.session_id:
            lines.append(f'Session: {ctx.session_id}')
        lines.append('')

        lines.append('## Session')
        if ctx.session_indexed:
            lines.append(f'Indexed: {ctx.session_chunks} chunks (transcript + session directory)')
            lines.append(
                'Search via mcp__document-search__search_documents (collection "document-chunks") '
                'to recover the matter(s) and their full reasoning.'
            )
        else:
            lines.append('(indexing skipped — search the existing index, or file from this conversation)')
        lines.append('')

        lines.append('## Docket — next NN per type')
        lines.append('')
        for type_name in DOCKET_TYPES:
            entries = ctx.existing[type_name]
            shown = ', '.join(entries) if entries else '(empty)'
            lines.append(f'  - docket/{type_name}/ — next: {ctx.next_ids[type_name]}  ·  existing: {shown}')

        return '\n'.join(lines)


if __name__ == '__main__':
    main()
