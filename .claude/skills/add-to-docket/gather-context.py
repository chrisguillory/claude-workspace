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
"""Index the session and resolve docket numbering for the add-to-docket skill."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import ClosedModel, SubsetModel
from document_search.schemas.indexing import IndexingResult

DOCKET = Path(__file__).resolve().parents[3] / 'docket'


class SessionInfo(SubsetModel):
    """Session metadata from `claude-session info`."""

    session_id: str
    session_file: str
    project_path: str


class TypeStatus(ClosedModel):
    """One docket type's numbering state: next NN, what's filed, and whether the dir is consistent."""

    name: str
    next_id: str
    existing: Sequence[str]
    duplicate: bool


class DocketContext(ClosedModel):
    """All gathered docket-filing context for one skill invocation."""

    session_id: str | None
    session_indexed: bool
    session_chunks: int
    types: Sequence[TypeStatus]
    requested: str | None


@ErrorBoundary(exit_code=1)
def main() -> None:
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    print(DocketContextGatherer(requested).run())


class DocketContextGatherer:
    """Index the session + resolve docket numbering for the add-to-docket skill."""

    def __init__(self, requested: str | None) -> None:
        self._requested = self._resolve_type(requested)

    def run(self) -> str:
        session_id, session_chunks = self._index_session()
        types = self._scan_docket()
        if self._requested:
            target = next(t for t in types if t.name == self._requested)
            if target.duplicate:
                raise ValueError(
                    f'docket/{target.name}/ already holds a duplicate NN — renumber it before filing '
                    f'(existing: {", ".join(target.existing) or "none"})'
                )
        ctx = DocketContext(
            session_id=session_id,
            session_indexed=session_chunks > 0,
            session_chunks=session_chunks,
            types=types,
            requested=self._requested,
        )
        return self._format_output(ctx)

    def _resolve_type(self, requested: str | None) -> str | None:
        """The human-named type (argv's first token), validated against the store's own types.

        Returns None when no recognized type was passed — bare invocation, a free-text
        description, or an un-interpolated arg — so the overview output still unblocks filing.
        """
        if not requested or not requested.split():
            return None
        first = requested.split()[0]
        valid = self._docket_types()
        if first in valid:
            return first
        print(f'(no docket type in {first!r}; pass one of: {", ".join(valid) or "—"})', file=sys.stderr)
        return None

    def _docket_types(self) -> Sequence[str]:
        """The store's self-declared types: every docket/<dir>/ that carries a README."""
        if not DOCKET.is_dir():
            return []
        return sorted(d.name for d in DOCKET.iterdir() if d.is_dir() and (d / 'README.md').is_file())

    def _scan_docket(self) -> Sequence[TypeStatus]:
        """Per type: next sequential NN (max existing + 1, 2-digit), existing entries, duplicate flag."""
        statuses: list[TypeStatus] = []
        for name in self._docket_types():
            entries = sorted(f.name for f in (DOCKET / name).glob('[0-9]*-*.md'))
            nums = [e.split('-', 1)[0] for e in entries]
            max_n = max((int(n) for n in nums if n.isdigit()), default=0)
            statuses.append(
                TypeStatus(
                    name=name,
                    next_id=f'{max_n + 1:02d}',
                    existing=entries,
                    duplicate=len(nums) != len(set(nums)),
                )
            )
        return statuses

    def _index_session(self) -> tuple[str | None, int]:
        """Index the session transcript + directory. Returns (session_id, total chunks).

        Degrades gracefully: a failed index (e.g. a transient embedding-provider overload) is
        surfaced on stderr and reported as 0 chunks, never silently. The session ID is still
        returned for the provenance footer.
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
            lines.append(f'Footer:  <sub>Claude Code session <code>{ctx.session_id}</code></sub>')
        lines.append('')

        lines.append('## Session')
        if ctx.session_indexed:
            lines.append(f'Indexed: {ctx.session_chunks} chunks (transcript + session directory)')
            lines.append(
                "Recover what you're filing + its full reasoning via mcp__document-search__search_documents "
                '(collection "document-chunks").'
            )
        else:
            lines.append('(indexing skipped — search the existing index, or file from this conversation)')
        lines.append('')

        lines.append('## Docket — next NN per type')
        if not ctx.types:
            lines.append('  (no docket types found — is the docket store present?)')
        for t in ctx.types:
            shown = ', '.join(t.existing) if t.existing else 'empty'
            warn = '   ⚠ duplicate NN — renumber before filing here' if t.duplicate else ''
            lines.append(f'  - {t.name}/ → docket/{t.name}/{t.next_id}-{{slug}}.md   ({shown}){warn}')
        lines.append('')

        if ctx.requested:
            lines.extend(self._resolve_block(ctx))
        else:
            lines.append('Pass a type for its path + filing guide: gather-context.py <type>')

        return '\n'.join(lines)

    def _resolve_block(self, ctx: DocketContext) -> Sequence[str]:
        """The targeted type's exact path + its pushed-out README."""
        target = next(t for t in ctx.types if t.name == ctx.requested)
        readme = (DOCKET / target.name / 'README.md').read_text().rstrip()
        rule = '─' * 70
        return [
            f'## Filing a {target.name} entry',
            f'Write to: docket/{target.name}/{target.next_id}-{{slug}}.md   — fill {{slug}} (kebab-case)',
            'Append the Footer (top). Follow the README below for what to capture.',
            '',
            rule,
            readme,
            rule,
        ]


if __name__ == '__main__':
    main()
