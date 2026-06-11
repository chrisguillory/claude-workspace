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
# cc-lib = { path = "../../../claude-workspace/cc-lib/", editable = true }
# document-search = { path = "../../../claude-workspace/mcp/document-search/", editable = true }
# ///
"""Gather session data, index artifacts, and output structured metadata.

Called by the /recover-session skill. Resolves session via `claude-session info`,
indexes the transcript and session directory via `document-search index`, analyzes the
transcript's record tree for forked/orphaned branches, recovers user messages sent
directly to subagent forks, and emits structured text optimized for model consumption.

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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pydantic
from cc_lib import ErrorBoundary
from cc_lib.schemas.base import SubsetModel
from document_search.schemas.indexing import IndexingResult
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELAY_TOOLS = ('SendMessage', 'Agent', 'Task')
MATCH_PREFIX = 110  # chars compared when matching a fork message against parent-relayed text
STALE_FORK_MIN_RECORDS = 10  # orphan size below which a late live sibling is not flagged
MIN_REPORTED_ORPHAN_RECORDS = 5  # smaller orphans without user messages roll up into a count
MESSAGE_PREVIEW_CHARS = 240
BRANCH_MESSAGE_CAP = 10

# ---------------------------------------------------------------------------
# Transcript-record vocabulary (subset of the JSONL we read; extras ignored)
# ---------------------------------------------------------------------------


class RecordOrigin(SubsetModel):
    """Message origin metadata (Claude Code 2.1.87+); presence ⇒ machine-relayed, not human.

    Subset of ``UserRecordOrigin``; ``kind`` is widened from the full Literal to ``str`` so
    a novel origin kind is read, not rejected.

    >>> # noinspection PyUnresolvedReferences
    >>> from claude_session.schemas.session.models import UserRecordOrigin
    """

    kind: str


class ContentBlock(SubsetModel):
    """One block of a message's content array (text / tool_use / thinking / image / ...).

    Subset of the content-block union; reads only the ``text`` and ``tool_use`` fields and
    keeps ``input`` untyped (the full per-tool input models are the fragile shape we skip).

    >>> # noinspection PyUnresolvedReferences
    >>> from claude_session.schemas.session.models import TextContent, ToolUseContent
    """

    type: str
    text: str | None = None
    name: str | None = None
    input: Mapping[str, object] | None = None


class RecordMessage(SubsetModel):
    """A record's message envelope; subset of ``Message``. ``content`` is a block list or a bare string.

    >>> # noinspection PyUnresolvedReferences
    >>> from claude_session.schemas.session.models import Message
    """

    content: Sequence[ContentBlock] | str = ()


class TranscriptRecord(SubsetModel):
    """The tree-walkable subset of a session record.

    Fields from ``BaseRecord`` (uuid/timestamp/type) and ``UserRecord`` (parentUuid/isMeta/
    origin/message; assistant records share the chain fields).

    ``uuid`` is None on summary/snapshot/queue records; ``parentUuid`` is None at chain
    roots (compaction restarts); ``origin``/``message`` are absent on non-message records.

    >>> # noinspection PyUnresolvedReferences
    >>> from claude_session.schemas.session.models import BaseRecord, UserRecord
    """

    type: str
    uuid: str | None = None
    parentUuid: str | None = None
    timestamp: str = ''
    isMeta: bool | None = None
    origin: RecordOrigin | None = None
    message: RecordMessage | None = None


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


class SessionData(SubsetModel):
    """Top-level container for all gathered session data."""

    session: SessionInfo
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

    Single pipeline: resolve session -> typed parse -> fork analysis -> scan directory
    -> index artifacts -> format output.
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
        fork_paths = _fork_transcripts(session_dir)
        fork_direct, fork_failures = _fork_direct_messages(records, fork_paths)

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


def _human_text(record: TranscriptRecord) -> str:
    """Human-typed text of a user record; '' for anything synthetic.

    Machine-relayed messages carry ``origin`` (task-notification, auto-continuation,
    channel, coordinator, peer; Claude Code 2.1.87+); older injected wrappers are filtered
    by prefix.
    """
    if record.type != 'user' or record.isMeta or record.origin is not None or record.message is None:
        return ''
    content = record.message.content
    text = content if isinstance(content, str) else '\n'.join(b.text or '' for b in content if b.type == 'text')
    text = text.strip()
    if not text or text.startswith(('<', 'Caveat:', '[Request interrupted')):
        return ''
    return text


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
                if (text := _human_text(nodes[u]))
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


def _fork_transcripts(session_dir: Path) -> Sequence[Path]:
    paths: list[Path] = []
    for directory in (session_dir, session_dir / 'subagents'):
        if directory.is_dir():
            paths.extend(directory.glob('agent-*.jsonl'))
    return sorted(set(paths))


def _fork_direct_messages(
    main_records: Sequence[TranscriptRecord], fork_paths: Sequence[Path]
) -> tuple[Sequence[ForkDirect], Mapping[str, int]]:
    """User messages typed directly to forks: not the spawn boilerplate, not parent-relayed."""
    if not fork_paths:
        return [], {}
    relayed = _parent_relayed(main_records)

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
            if (text := _human_text(record)) and '<fork-boilerplate>' not in text and not _is_relayed(text, relayed)
        ]
        if direct:
            results.append(ForkDirect(fork_name=name, messages=direct))
    return results, dict(failures)


def _parent_relayed(records: Sequence[TranscriptRecord]) -> Sequence[str]:
    """Normalized text of every directive the parent relayed (SendMessage / Agent / Task tool calls)."""
    relayed: list[str] = []
    for record in records:
        content = record.message.content if record.message else None
        if not isinstance(content, Sequence) or isinstance(content, str):
            continue
        for block in content:
            if block.type != 'tool_use' or block.name not in RELAY_TOOLS or block.input is None:
                continue
            for key in ('content', 'message', 'prompt'):
                value = block.input.get(key)
                if isinstance(value, str) and len(value) > 30:
                    relayed.append(_normalize(value))
    return relayed


def _is_relayed(text: str, relayed: Iterable[str]) -> bool:
    normalized = _normalize(text)
    head = normalized[:MATCH_PREFIX]
    return any(head in candidate or candidate[:MATCH_PREFIX] in normalized for candidate in relayed)


# ---------------------------------------------------------------------------
# Private helpers — text shaping
# ---------------------------------------------------------------------------


def _preview(text: str) -> str:
    flattened = re.sub(r'\s+', ' ', text).strip()
    return flattened[:MESSAGE_PREVIEW_CHARS] + ('…' if len(flattened) > MESSAGE_PREVIEW_CHARS else '')


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


if __name__ == '__main__':
    main()
