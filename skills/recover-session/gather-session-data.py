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
# cc-lib = { path = "../../cc-lib/", editable = true }
# document-search = { path = "../../mcp/document-search/", editable = true }
# ///
"""Gather session data, index artifacts, and output structured metadata.

Called by the /recover-session skill. Resolves session via `claude-session info`,
orients against deterministic git ground-truth (every worktree's branch/status, with deep
detail for the work-candidate tree(s)), indexes the transcript and session directory via
`document-search index`, analyzes the transcript's record tree for forked/orphaned branches,
recovers user messages sent directly to subagent forks, and emits structured text optimized
for model consumption.

PARSING — local SubsetModel vocabulary, not the full claude-session schema
--------------------------------------------------------------------------
Records are parsed into a local `TranscriptRecord` (cc-lib `SubsetModel`, extra='ignore')
covering only the seven stable fields the analysis reads: uuid, parentUuid, timestamp,
type, isMeta, origin.kind, message.content. This is typed validation, not raw-dict access.

The full claude-session schema is deliberately not imported here. A read-only analyzer
round-trips nothing, so the strict-everywhere doctrine (which exists so clone/archive/
restore don't corrupt what they don't model) does not apply. Worse, full strict models
*drop* any record carrying a field/enum/record-type the installed schema predates — and a
dropped mid-chain record severs the parentUuid linkage, silently halving the recovery view
(measured: one drifted mid-chain record cut the live walk 680->340 and forged a phantom
generation). Recovery runs disproportionately right after a Claude Code release, before the
schema is bumped — exactly when that happens. The subset reads only the format's most
fossilized fields and never touches the fragile ones (Literals, attachment/tool shapes), so
it survives drift; genuine core-field drift still surfaces as a counted parse failure.

FORK ANALYSIS (the recoverables a mainline walk never sees)
------------------------------------------------------------
Transcript records form a tree chained by ``uuid -> parentUuid``; a resume rebuilds
context by walking back from the file's newest leaf. Two classes of data fall off that
walk:

1. *Orphaned sibling branches* — rewinds, and the stale-fork bug: a still-open instance
   flushes its pending Esc-interrupt records hours late, forking the tree after the fact
   and stealing the resume pointer (signature: the live-chain child postdates the entire
   orphaned branch). The orphan's user messages — and sometimes the real work — never
   replay into the resumed context.
2. *Fork-direct user messages* — the user can type straight to a background fork; those
   messages live only in the fork's ``agent-*.jsonl``, never the main transcript. Per
   fork, a user-role message that is neither the ``<fork-boilerplate>`` spawn directive
   nor matched (first ~110 normalized chars) by a parent-relayed SendMessage/Agent/Task
   tool input is the user's direct message.

Compaction restarts the chain at a new root (``parentUuid=null``); earlier generations
are reported as preserved history, not loss. Single-record micro-orphans (message edits,
retries) are rolled up into a count rather than listed.

Usage:
    gather-session-data.py [SESSION_ID]

If SESSION_ID is omitted, auto-detects the current session via claude-session.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pydantic
from cc_lib import ErrorBoundary
from cc_lib.schemas.base import ClosedModel, SubsetModel
from cc_lib.transcript_spine import TranscriptRecord, fork_transcripts, human_text, is_relayed, parent_relayed
from document_search.schemas.indexing import IndexingResult
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STALE_FORK_MIN_RECORDS = 10  # orphan size below which a late live sibling is not flagged
MIN_REPORTED_ORPHAN_RECORDS = 5  # smaller orphans without user messages roll up into a count
MESSAGE_PREVIEW_CHARS = 240
BRANCH_MESSAGE_CAP = 10
RECENT_COMMITS_SHOWN = 10  # `git log --oneline -N` depth in the deep detail for work-candidate trees

# Transcript records + the human/queued/fork directive filter come from cc_lib.transcript_spine.


# ---------------------------------------------------------------------------
# Output models (vocabulary emitted to the skill prompt)
# ---------------------------------------------------------------------------


class SessionInfo(SubsetModel):
    """Session metadata from claude-session info --format json."""

    session_id: str
    custom_title: str | None = None
    project_path: str
    session_file: str
    source: str | None = None
    state: str | None = None
    first_message_at: str | None = None
    parent_id: str | None = None


class UserMessage(BaseModel):
    """A recovered human message with its timestamp."""

    timestamp: str
    text: str


class OrphanBranch(BaseModel):
    """A branch of the live tree that the resume walk does not reach."""

    fork_uuid: str  # shared ancestor where the branch diverges from the live chain
    leaf_uuid: str  # branch tip (locator for reading the branch back in the JSONL)
    record_count: int
    first_at: str
    last_at: str
    stale_fork_suspected: bool  # live sibling written after this branch's tip
    user_messages: Sequence[UserMessage]


class TreeAnalysis(BaseModel):
    """Fork/orphan analysis of the main transcript's record tree."""

    total_records: int
    tree_records: int  # records carrying a uuid (tree-walkable)
    generation_count: int  # roots: the live tree + prior compaction generations
    live_chain_records: int
    orphan_branches: Sequence[OrphanBranch]  # reported branches (human messages or substantial)
    micro_orphan_count: int  # rolled-up small branches without user messages

    @property
    def stale_fork(self) -> bool:
        return any(b.stale_fork_suspected for b in self.orphan_branches)


class ForkDirect(BaseModel):
    """User messages typed directly to one subagent fork."""

    fork_name: str
    messages: Sequence[UserMessage]


class WorktreeStatus(ClosedModel):
    """One git worktree's deterministic ground-truth for recovery orientation."""

    path: str
    branch: str  # branch name, or '(detached)'
    head_short: str
    dirty: bool
    ahead_of_trunk: int  # commits on HEAD not on trunk (main/master); 0 if no trunk
    locked: bool
    is_session_cwd: bool
    last_commit_subject: str

    @property
    def is_work_candidate(self) -> bool:
        """Dirty or ahead of trunk ⇒ likely holds in-progress work (not a clean trunk checkout)."""
        return self.dirty or self.ahead_of_trunk > 0


class WorktreeDetail(ClosedModel):
    """Deep git state for a work-candidate tree (or the session-cwd tree when none qualify).

    Commits are windowed to the session's lifetime (``git log --since=first_message_at``) so
    they answer "what did THIS session do", not "what are the last N commits"; an absent
    anchor falls back to a fixed ``-N`` count. The blobs are raw git output, emitted verbatim.
    """

    path: str
    branch: str
    recent_commits: Sequence[str]  # `git log --oneline` lines, session-windowed (may be empty)
    commits_session_windowed: bool  # True ⇒ since session start; False ⇒ fixed -N fallback
    status_short: Sequence[str]  # `git status --short` lines (may be empty)
    diff_stat: str  # `git diff --stat` summary line (uncommitted changes); '' if clean
    stashes: Sequence[str]  # `git stash list` lines (may be empty)


class SessionData(SubsetModel):
    """Top-level container for all gathered session data."""

    session: SessionInfo
    session_cwd: str | None
    worktrees: Sequence[WorktreeStatus]
    worktree_details: Sequence[WorktreeDetail]
    transcript_size_mb: float
    transcript_lines: int
    transcript_parse_failures: Mapping[str, int]
    session_dir_path: str | None
    session_dir_file_count: int
    session_dir_subdirs: Sequence[str]
    agent_transcript_count: int
    tree: TreeAnalysis | None
    fork_direct: Sequence[ForkDirect]
    fork_transcript_count: int
    fork_parse_failures: Mapping[str, int]
    transcript_index: IndexingResult | None
    session_dir_index: IndexingResult | None
    html_export_path: str | None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


@ErrorBoundary(exit_code=1)
def main() -> None:
    """Entry point with error boundary."""
    session_id = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else None
    gatherer = SessionGatherer()
    code = gatherer.run(session_id)
    if code != 0:
        sys.exit(code)


# ---------------------------------------------------------------------------
# Primary abstraction
# ---------------------------------------------------------------------------


class SessionGatherer:
    """Resolve, analyze, index, and report on a Claude Code session.

    Single pipeline: resolve session -> typed parse -> fork analysis -> git orientation
    -> scan directory -> index artifacts -> format output.
    """

    def run(self, session_id: str | None = None) -> int:
        """Entry point. Returns 0 on success, 1 on failure."""
        session = self._resolve_session(session_id)
        if not session:
            if session_id:
                print(f'Could not resolve session: {session_id}', file=sys.stderr)
            else:
                print('Could not auto-detect session. Provide a session ID:', file=sys.stderr)
                print('  /recover-session <session-id>', file=sys.stderr)
            return 1

        data = self._gather(session)
        print(self._format_output(data))
        return 0

    def _resolve_session(self, session_id: str | None) -> SessionInfo | None:
        """Resolve session via claude-session info CLI."""
        cmd = ['claude-session', 'info', '--format', 'json']
        if session_id:
            cmd.insert(2, session_id)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f'claude-session info failed: {result.stderr.strip()}', file=sys.stderr)
            return None

        return SessionInfo.model_validate(json.loads(result.stdout))

    def _gather(self, session: SessionInfo) -> SessionData:
        """Gather all session data: metadata, fork analysis, indexing, directory contents."""
        transcript = Path(session.session_file)
        session_dir = transcript.parent / session.session_id
        html_export = transcript.parent / f'{session.session_id}.html'

        # Transcript metadata
        transcript_size_mb = round(transcript.stat().st_size / (1024 * 1024), 1)
        with open(transcript) as f:
            transcript_lines = sum(1 for _ in f)

        # Session directory metadata
        session_dir_exists = session_dir.is_dir()
        session_dir_file_count = 0
        session_dir_subdirs: list[str] = []
        agent_transcript_count = 0

        if session_dir_exists:
            for entry in session_dir.iterdir():
                if entry.is_file():
                    session_dir_file_count += 1
                    if entry.name.startswith('agent-') and entry.suffix == '.jsonl':
                        agent_transcript_count += 1
                elif entry.is_dir():
                    session_dir_subdirs.append(entry.name)
                    session_dir_file_count += sum(1 for f in entry.rglob('*') if f.is_file())

        # Typed parse + fork analysis (before indexing so they still emit if indexing fails)
        print('Parsing transcript (subset schema) and analyzing record tree...', file=sys.stderr)
        records, parse_failures = _parse_records(transcript)
        tree = _analyze_tree(records)
        fork_paths = fork_transcripts(session_dir)
        fork_direct, fork_failures = _fork_direct_messages(records, fork_paths)

        # Git ground-truth (fast, deterministic) — every worktree's branch + status, plus
        # deep detail for the work-candidate tree(s) (session-cwd tree if none qualify).
        print('Orienting git worktrees...', file=sys.stderr)
        session_cwd = self._session_cwd(session)
        worktrees = self._enumerate_worktrees(session_cwd) if session_cwd else []
        worktree_details = self._worktree_details(session, worktrees)

        # Index transcript
        print(f'Indexing transcript ({transcript_size_mb} MB, {transcript_lines} lines)...', file=sys.stderr)
        transcript_index = self._index_path(str(transcript))

        # Index session directory
        session_dir_index = None
        if session_dir_exists:
            print(f'Indexing session directory ({session_dir_file_count} files)...', file=sys.stderr)
            session_dir_index = self._index_path(str(session_dir))

        return SessionData(
            session=session,
            session_cwd=str(session_cwd) if session_cwd else None,
            worktrees=worktrees,
            worktree_details=worktree_details,
            transcript_size_mb=transcript_size_mb,
            transcript_lines=transcript_lines,
            transcript_parse_failures=parse_failures,
            session_dir_path=str(session_dir) if session_dir_exists else None,
            session_dir_file_count=session_dir_file_count,
            session_dir_subdirs=session_dir_subdirs,
            agent_transcript_count=agent_transcript_count,
            tree=tree,
            fork_direct=fork_direct,
            fork_transcript_count=len(fork_paths),
            fork_parse_failures=fork_failures,
            transcript_index=transcript_index,
            session_dir_index=session_dir_index,
            html_export_path=str(html_export) if html_export.is_file() else None,
        )

    def _index_path(self, path: str) -> IndexingResult | None:
        """Index a path via document-search index CLI."""
        result = subprocess.run(
            ['document-search', 'index', path, '-c', 'document-chunks', '--no-gitignore', '--format', 'json'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f'Index failed for {path}: {result.stderr.strip()}', file=sys.stderr)
            return None

        return IndexingResult.model_validate(json.loads(result.stdout))

    def _git(self, cwd: Path | str, *args: str) -> str | None:
        """Run a git subcommand in `cwd`; return stripped stdout, or None on non-zero exit."""
        result = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    def _session_cwd(self, session: SessionInfo) -> Path | None:
        """The session's launch cwd, scanned from the transcript (records carry `cwd`).

        Anchors the worktree enumeration — any cwd inside the repo works, since
        `git worktree list` returns every linked worktree. Best-effort: None if unfound.
        """
        with open(session.session_file) as f:
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                if cwd := TranscriptRecord.model_validate_json(line).cwd:
                    return Path(cwd)
        return None

    def _enumerate_worktrees(self, session_cwd: Path) -> Sequence[WorktreeStatus]:
        """Every linked worktree's branch + status. Best-effort: empty if not a git repo.

        Worktrees nest (linked trees live under the primary clone), so the session cwd is
        attributed to its *deepest* containing worktree — the launch tree, not the parent.
        """
        porcelain = self._git(session_cwd, 'worktree', 'list', '--porcelain')
        if not porcelain:
            return []
        blocks = [dict(self._parse_porcelain_block(b)) for b in porcelain.split('\n\n')]
        blocks = [b for b in blocks if b.get('worktree')]
        trunk = self._detect_trunk(session_cwd)
        containers = [
            b['worktree']
            for b in blocks
            if session_cwd == Path(b['worktree']) or Path(b['worktree']) in session_cwd.parents
        ]
        cwd_worktree = max(containers, key=len, default=None)

        worktrees: list[WorktreeStatus] = []
        for block in blocks:
            path = block['worktree']
            ahead = 0
            if trunk:
                count = self._git(path, 'rev-list', '--count', f'{trunk}..HEAD')
                ahead = int(count) if count and count.isdigit() else 0
            worktrees.append(
                WorktreeStatus(
                    path=path,
                    branch=block['branch'].removeprefix('refs/heads/') if 'branch' in block else '(detached)',
                    head_short=block.get('HEAD', '')[:8],
                    dirty=bool(self._git(path, 'status', '--porcelain')),
                    ahead_of_trunk=ahead,
                    locked='locked' in block,
                    is_session_cwd=(path == cwd_worktree),
                    last_commit_subject=self._git(path, 'log', '-1', '--format=%s') or '',
                )
            )
        return worktrees

    def _worktree_details(self, session: SessionInfo, worktrees: Sequence[WorktreeStatus]) -> Sequence[WorktreeDetail]:
        """Deep git state for the work-candidate tree(s); the session-cwd tree if none qualify.

        Clean siblings are left out — only trees likely to own in-progress work get the
        commit/status/stash dump. With no work candidate at all, the session-cwd tree is the
        single best fallback to orient against.
        """
        targets = [w for w in worktrees if w.is_work_candidate] or [w for w in worktrees if w.is_session_cwd]
        return [self._collect_detail(w, session.first_message_at) for w in targets]

    def _collect_detail(self, worktree: WorktreeStatus, session_start: str | None) -> WorktreeDetail:
        """One tree's commit window, working-tree status, diff-stat, and stashes."""
        windowed = session_start is not None
        log_args = (
            ['log', '--oneline', f'--since={session_start}']
            if windowed
            else ['log', '--oneline', f'-{RECENT_COMMITS_SHOWN}']
        )
        commits = self._git(worktree.path, *log_args) or ''
        status = self._git(worktree.path, 'status', '--short') or ''
        diff_stat = self._git(worktree.path, 'diff', '--stat') or ''
        stashes = self._git(worktree.path, 'stash', 'list') or ''
        return WorktreeDetail(
            path=worktree.path,
            branch=worktree.branch,
            recent_commits=commits.splitlines(),
            commits_session_windowed=windowed,
            status_short=status.splitlines(),
            diff_stat=diff_stat.splitlines()[-1].strip() if diff_stat else '',
            stashes=stashes.splitlines(),
        )

    def _detect_trunk(self, cwd: Path) -> str | None:
        """The mainline branch for ahead-counts — local `main`/`master` if present."""
        for ref in ('main', 'master'):
            if self._git(cwd, 'rev-parse', '--verify', '--quiet', ref) is not None:
                return ref
        return None

    @staticmethod
    def _parse_porcelain_block(block: str) -> Iterator[tuple[str, str]]:
        """Yield (key, value) for each line of a `git worktree list --porcelain` record."""
        for line in block.splitlines():
            if line.strip():
                key, _, value = line.partition(' ')
                yield key, value

    def _format_output(self, data: SessionData) -> str:
        """Format gathered data as structured text for the skill prompt."""
        lines: list[str] = []
        s = data.session

        lines.append(f'Session: {s.session_id}')
        if s.custom_title:
            lines.append(f'Title: {s.custom_title}')
        lines.append(f'Project: {s.project_path}')
        if s.source:
            lines.append(f'Source: {s.source}')
        if s.state:
            lines.append(f'State: {s.state}')
        if s.first_message_at:
            lines.append(f'Started: {s.first_message_at}')
        lines.append('')

        lines.extend(self._format_git_orientation(data))

        # Transcript
        lines.append('## Transcript')
        lines.append(f'Path: {s.session_file}')
        lines.append(f'Size: {data.transcript_size_mb} MB ({data.transcript_lines} lines)')
        if data.transcript_parse_failures:
            lines.append(
                f'Subset-parse FAILURES (core-field drift on uuid/parentUuid/type/...): '
                f'{dict(data.transcript_parse_failures)}'
            )
        if data.transcript_index:
            idx = data.transcript_index
            total = idx.chunks_created + idx.chunks_skipped
            cached = f', {idx.files_cached} cached' if idx.files_cached else ''
            lines.append(f'Indexed: {total} chunks ({idx.elapsed_seconds:.1f}s{cached})')
        else:
            lines.append('Index: FAILED (see stderr)')
        lines.append('')

        lines.extend(self._format_tree(data.tree))
        lines.extend(self._format_fork_direct(data.fork_direct, data.fork_transcript_count, data.fork_parse_failures))

        # Session directory
        lines.append('## Session directory')
        if data.session_dir_path:
            lines.append(f'Path: {data.session_dir_path}')
            lines.append(f'Files: {data.session_dir_file_count}')
            if data.session_dir_subdirs:
                lines.append(f'Subdirectories: {", ".join(data.session_dir_subdirs)}')
            if data.agent_transcript_count:
                lines.append(f'Agent transcripts: {data.agent_transcript_count}')
            if data.session_dir_index:
                sidx = data.session_dir_index
                total = sidx.chunks_created + sidx.chunks_skipped
                cached = f', {sidx.files_cached} cached' if sidx.files_cached else ''
                lines.append(f'Indexed: {total} chunks ({sidx.elapsed_seconds:.1f}s{cached})')
            elif data.session_dir_file_count:
                lines.append('Index: FAILED (see stderr)')
        else:
            lines.append('(not found)')
        lines.append('')

        # HTML export
        if data.html_export_path:
            lines.append('## HTML export')
            lines.append(f'Path: {data.html_export_path}')
            lines.append('')

        return '\n'.join(lines)

    def _format_tree(self, tree: TreeAnalysis | None) -> Sequence[str]:
        lines = ['## Fork analysis (main transcript)']
        if tree is None:
            lines.extend(['(no tree-walkable records)', ''])
            return lines

        lines.append(
            f'Records: {tree.total_records} ({tree.tree_records} tree-walkable) | '
            f'generations (compaction restarts): {tree.generation_count} | '
            f'live chain: {tree.live_chain_records} records'
        )
        if tree.stale_fork:
            lines.append(
                '!! STALE FORK SUSPECTED — the live chain diverged AFTER an orphaned branch completed. '
                'The resume may be anchored on a dead twig (e.g. a stale instance flushed an Esc-interrupt '
                'late); the orphaned branch below is likely the real continuation.'
            )
        if not tree.orphan_branches and not tree.micro_orphan_count:
            lines.extend(['Live tree has no orphaned branches.', ''])
            return lines

        if tree.orphan_branches:
            lines.append(f'Orphaned branches on the live tree: {len(tree.orphan_branches)}')
            for branch in tree.orphan_branches:
                flag = '  [STALE-FORK SUSPECT]' if branch.stale_fork_suspected else ''
                lines.append(
                    f'- {branch.record_count} records | {branch.first_at} -> {branch.last_at} | '
                    f'fork @ {branch.fork_uuid[:8]} | leaf {branch.leaf_uuid}{flag}'
                )
                if branch.user_messages:
                    shown = branch.user_messages[:BRANCH_MESSAGE_CAP]
                    lines.append(f'  user messages ({len(shown)} of {len(branch.user_messages)}):')
                    lines.extend(f'    [{message.timestamp}] {message.text}' for message in shown)
                else:
                    lines.append('  user messages: none')
            lines.append(
                'Orphan content is in the semantic index — search it (or read the JSONL at a leaf uuid) '
                'and fold it into the recovered state.'
            )
        if tree.micro_orphan_count:
            lines.append(
                f'Micro-orphans rolled up: {tree.micro_orphan_count} '
                f'(< {MIN_REPORTED_ORPHAN_RECORDS} records, no user messages — edits/retries)'
            )
        lines.append('')
        return lines

    def _format_fork_direct(
        self, fork_direct: Sequence[ForkDirect], fork_count: int, fork_failures: Mapping[str, int]
    ) -> Sequence[str]:
        lines = ['## Fork-direct user messages (subagents)']
        if fork_failures:
            lines.append(f'Subset-parse FAILURES in fork transcripts: {dict(fork_failures)}')
        if not fork_count:
            lines.extend(['(no fork transcripts)', ''])
            return lines
        if not fork_direct:
            lines.extend([f'None found across {fork_count} fork transcript(s).', ''])
            return lines

        for fork in fork_direct:
            lines.append(f'- {fork.fork_name}: {len(fork.messages)} direct message(s)')
            lines.extend(f'    [{message.timestamp}] {message.text}' for message in fork.messages)
        lines.append('')
        return lines

    def _format_git_orientation(self, data: SessionData) -> Sequence[str]:
        """Deterministic git ground-truth: a worktree table, then deep detail for ★ trees."""
        lines = ['## Git orientation']
        if not data.worktrees:
            lines.extend(['(no git worktrees — session cwd not in a repo)', ''])
            return lines

        if data.session_cwd:
            lines.append(f'Session cwd: {data.session_cwd}')
        lines.append(
            '★ = likely holds in-progress work (dirty or ahead of trunk) — reconcile these against '
            "the summary's claimed edits, and run the edit audit in the tree that owns them."
        )
        for w in data.worktrees:
            marker = '★' if w.is_work_candidate else ' '
            status_bits = ([f'+{w.ahead_of_trunk} ahead'] if w.ahead_of_trunk else []) + (['dirty'] if w.dirty else [])
            flags = (['← session cwd'] if w.is_session_cwd else []) + (['locked'] if w.locked else [])
            flag_str = f'  [{"; ".join(flags)}]' if flags else ''
            status = ', '.join(status_bits) or 'clean'
            lines.append(f'{marker} {w.branch} @{w.head_short}  {status}{flag_str}')
            row = f'    {w.path}'
            if w.is_work_candidate and w.last_commit_subject:
                row += f'  — "{w.last_commit_subject}"'
            lines.append(row)
        lines.append('')

        for detail in data.worktree_details:
            lines.extend(self._format_worktree_detail(detail))
        return lines

    def _format_worktree_detail(self, detail: WorktreeDetail) -> Sequence[str]:
        """One work-candidate tree's commits, working-tree status, diff-stat, and stashes."""
        lines = [f'### {detail.branch} — {detail.path}']
        if detail.commits_session_windowed:
            header, empty = 'Commits since session start:', '  (no commits since session start)'
        else:
            header, empty = f'Recent commits (no session-start anchor; last {RECENT_COMMITS_SHOWN}):', '  (no commits)'
        lines.append(header)
        if detail.recent_commits:
            lines.extend(f'  {c}' for c in detail.recent_commits)
        else:
            lines.append(empty)
        if detail.diff_stat:
            lines.append(f'Uncommitted: {detail.diff_stat}')
        if detail.status_short:
            lines.append('Working-tree status:')
            lines.extend(f'  {s}' for s in detail.status_short)
        else:
            lines.append('Working tree: clean')
        if detail.stashes:
            lines.append('Stashes:')
            lines.extend(f'  {s}' for s in detail.stashes)
        lines.append('')
        return lines


# ---------------------------------------------------------------------------
# Private helpers — typed parsing
# ---------------------------------------------------------------------------


def _parse_records(path: Path) -> tuple[Sequence[TranscriptRecord], Mapping[str, int]]:
    """Parse a JSONL transcript into the TranscriptRecord subset.

    Recovery must survive schema drift, so per-record validation failures are counted by
    record type (and surfaced in the report) instead of aborting. With the subset only the
    seven core fields can fail, so a failure here is genuine core-field drift that would
    break the tree walk — exactly the signal worth surfacing.
    """
    records: list[TranscriptRecord] = []
    failures: defaultdict[str, int] = defaultdict(int)
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                failures['<unparseable-json>'] += 1
                continue
            try:
                records.append(TranscriptRecord.model_validate(raw))
            except pydantic.ValidationError:
                failures[str(raw.get('type'))] += 1
    return records, dict(failures)


# ---------------------------------------------------------------------------
# Private helpers — main-transcript tree analysis
# ---------------------------------------------------------------------------


def _analyze_tree(records: Sequence[TranscriptRecord]) -> TreeAnalysis | None:
    """Build the uuid->parentUuid tree and report orphaned branches off the live chain.

    The live chain is the ancestry of the file's newest record — what a resume anchors
    to. Branches forking off it are orphans; separate roots are compaction generations
    (preserved history, not loss) and are only counted.
    """
    nodes: dict[str, TranscriptRecord] = {}
    order: list[str] = []
    for record in records:
        if record.uuid is not None:
            nodes[record.uuid] = record
            order.append(record.uuid)
    if not order:
        return None

    children: defaultdict[str, list[str]] = defaultdict(list)
    root_count = 0
    for uuid in order:
        parent = nodes[uuid].parentUuid
        if parent and parent in nodes:
            children[parent].append(uuid)
        else:
            root_count += 1

    # Live chain: walk back from the newest record.
    live: list[str] = []
    seen: set[str] = set()
    cursor: str | None = order[-1]
    while cursor and cursor in nodes and cursor not in seen:
        live.append(cursor)
        seen.add(cursor)
        cursor = nodes[cursor].parentUuid
    live_set = set(live)

    reported: list[OrphanBranch] = []
    micro_count = 0
    for ancestor in live:
        live_child = next((c for c in children.get(ancestor, []) if c in live_set), None)
        for child in children.get(ancestor, []):
            if child in live_set:
                continue
            branch = _collect_subtree(child, children)
            timestamps = sorted(nodes[u].timestamp for u in branch)
            first_at, last_at = timestamps[0], timestamps[-1]
            messages = [
                UserMessage(timestamp=nodes[u].timestamp, text=_preview(text))
                for u in branch
                if (text := human_text(nodes[u]))
            ]
            if not messages and len(branch) < MIN_REPORTED_ORPHAN_RECORDS:
                micro_count += 1
                continue
            stale = bool(
                live_child
                and nodes[live_child].timestamp
                and last_at
                and nodes[live_child].timestamp > last_at
                and len(branch) >= STALE_FORK_MIN_RECORDS
            )
            reported.append(
                OrphanBranch(
                    fork_uuid=ancestor,
                    leaf_uuid=max(branch, key=lambda u: nodes[u].timestamp),
                    record_count=len(branch),
                    first_at=first_at,
                    last_at=last_at,
                    stale_fork_suspected=stale,
                    user_messages=messages,
                )
            )

    reported.sort(key=lambda b: b.record_count, reverse=True)
    return TreeAnalysis(
        total_records=len(records),
        tree_records=len(order),
        generation_count=root_count,
        live_chain_records=len(live),
        orphan_branches=reported,
        micro_orphan_count=micro_count,
    )


def _collect_subtree(root: str, children: Mapping[str, Sequence[str]]) -> Sequence[str]:
    collected: list[str] = []
    stack = [root]
    while stack:
        uuid = stack.pop()
        collected.append(uuid)
        stack.extend(children.get(uuid, []))
    return collected


# ---------------------------------------------------------------------------
# Private helpers — fork-direct user messages
# ---------------------------------------------------------------------------


def _fork_direct_messages(
    main_records: Sequence[TranscriptRecord], fork_paths: Sequence[Path]
) -> tuple[Sequence[ForkDirect], Mapping[str, int]]:
    """User messages typed directly to forks: not the spawn boilerplate, not parent-relayed."""
    if not fork_paths:
        return [], {}
    relayed = parent_relayed(main_records)

    results: list[ForkDirect] = []
    failures: defaultdict[str, int] = defaultdict(int)
    for fork_path in fork_paths:
        name = fork_path.name.removeprefix('agent-').removesuffix('.jsonl')
        fork_records, fork_failures = _parse_records(fork_path)
        for record_type, count in fork_failures.items():
            failures[record_type] += count
        direct = [
            UserMessage(timestamp=record.timestamp, text=_preview(text))
            for record in fork_records
            if (text := human_text(record)) and not is_relayed(text, relayed)
        ]
        if direct:
            results.append(ForkDirect(fork_name=name, messages=direct))
    return results, dict(failures)


# ---------------------------------------------------------------------------
# Private helpers — text shaping
# ---------------------------------------------------------------------------


def _preview(text: str) -> str:
    flattened = re.sub(r'\s+', ' ', text).strip()
    return flattened[:MESSAGE_PREVIEW_CHARS] + ('…' if len(flattened) > MESSAGE_PREVIEW_CHARS else '')


if __name__ == '__main__':
    main()
