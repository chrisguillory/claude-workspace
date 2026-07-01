"""Claude Code session transcript parsing + user-intent spine extraction.

The user-intent spine is the deterministic record of every directive the user gave in a
session — the load-bearing signal a summary rounds off. A directive reaches the transcript
three ways, and a faithful spine captures all three:

- a normal user turn          — ``type='user'``; origin kind ``'human'`` (2.1.x+) or absent (older)
- a directive queued mid-work — an attachment of type ``'queued_command'`` (its ``prompt``)
- a directive typed to a fork — a human message in a subagent transcript (``agent-*.jsonl``)

Machine-relayed user records (origin kinds task-notification / auto-continuation / channel /
coordinator / peer; 2.1.87+) and injected scaffolding (skill headers, local-command I/O, compaction
summaries, and Claude-Code-injected fork spawn prompts) are not directives, and are dropped.

This is the single home for the filter — create-pr and recover-session consume it; the where-am-i
skill converges onto it once it lands. Regression-tested: a 2.1.x ``origin.kind=='human'`` record
must survive, and CC-injected fork spawn prompts must not leak in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence, Set
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pydantic

from cc_lib.schemas.base import ClosedModel, SubsetModel

__all__ = [
    'ContentBlock',
    'Message',
    'Origin',
    'QueuedCommand',
    'SpineEntry',
    'TranscriptRecord',
    'extract_spine',
    'fork_transcripts',
    'human_text',
    'is_relayed',
    'parent_relayed',
    'parse_transcript',
    'queued_text',
]

PACIFIC = ZoneInfo('America/Los_Angeles')
# Tools whose calls relay a directive into a fork — their inputs are the orchestrator's words, not the user's.
RELAY_TOOLS = {'Task', 'Agent', 'SendMessage'}
MATCH_PREFIX = 110  # prefix length for relay matching (a relayed prompt may be lightly reworded)
# A directive's text starting with any of these is injected scaffolding, not something the user typed.
SKIP_PREFIXES = (
    '<',  # task-notification / local-command / bash-io / fork-boilerplate wrappers
    'Caveat:',
    '[Request interrupted',
    'This session is being continued from a previous',
    'Base directory for this skill',
    'Your task is to create a detailed summary of the conversation',  # CC compaction-fork spawn prompt
    'CRITICAL: Respond with TEXT ONLY',  # CC probe-fork spawn prompt
    '[SUGGESTION MODE',  # CC autocomplete-fork spawn prompt
)
# CC fork prompts whose entire message is a fixed token — unsafe as a prefix (would drop a real
# "Warmup the cache" directive), so matched exactly instead.
SKIP_EXACT: Set[str] = {'Warmup'}  # CC cache-warmup fork


class Origin(ClosedModel):
    """A message's origin tag; kind 'human' marks a real user directive, other kinds are machine-relayed."""

    kind: str | None = None


class ContentBlock(SubsetModel):
    """A message content block — only the fields the spine needs (text + relay tool calls)."""

    type: str | None = None
    text: str | None = None
    name: str | None = None
    input: Mapping[str, object] | None = None


class Message(SubsetModel):
    """A record's message envelope; content is a string or a list of blocks."""

    content: str | Sequence[ContentBlock] | None = None


class QueuedCommand(SubsetModel):
    """A ``queued_command`` attachment — a directive the user typed while the agent was working."""

    type: str | None = None
    prompt: str | Sequence[ContentBlock] | None = (
        None  # wire shape: text | (TextContent|ImageContent)[], not bare strings
    )
    origin: Origin | None = None


class TranscriptRecord(SubsetModel):
    """A Claude Code transcript record — the shared subset every spine/recovery consumer reads.

    Directive fields (type/isMeta/origin/message/attachment + timestamp) drive spine extraction;
    the tree/launch fields (uuid/parentUuid/cwd) serve recover-session's fork-and-orphan analysis.
    """

    type: str | None = None
    uuid: str | None = None
    parentUuid: str | None = None  # noqa: N815 — wire field; recover-session walks the uuid→parentUuid tree
    timestamp: str = ''  # present on every message/event record; '' only on the rare timestamp-less record
    isMeta: bool | None = None  # noqa: N815 — wire field name (Claude Code camelCase)
    origin: Origin | None = None  # kind=='human' ⇒ user; other kinds ⇒ machine-relayed (CC 2.1.x+)
    message: Message | None = None
    attachment: QueuedCommand | None = None
    cwd: str | None = None  # launch directory (BaseRecord.cwd); recover-session anchors worktree enumeration on it


class SpineEntry(ClosedModel):
    """One directive in the spine, with its Pacific-rendered timestamp and where it came from."""

    timestamp: str
    text: str
    source: str  # 'user' | 'queued' | 'fork:<name>'


def parse_transcript(path: Path) -> Sequence[TranscriptRecord]:
    """Parse a transcript JSONL into records, skipping malformed lines (drift-tolerant)."""
    records: list[TranscriptRecord] = []
    with path.open(errors='ignore') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TranscriptRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, pydantic.ValidationError):
                continue
    return records


def human_text(record: TranscriptRecord) -> str:
    """Human-typed text of a normal user turn; '' for anything synthetic.

    A real user message either has no ``origin`` (older Claude Code) or ``origin.kind == 'human'``
    (2.1.x+, which tags every user record); machine-relayed messages carry other kinds.
    """
    if record.type != 'user' or record.isMeta or record.message is None or not _is_human(record.origin):
        return ''
    content = record.message.content
    if isinstance(content, str):
        text = content
    elif content is None:
        return ''
    else:
        text = '\n'.join(b.text or '' for b in content if b.type == 'text')
    return _clean(text)


def queued_text(record: TranscriptRecord) -> str:
    """Directive text of a ``queued_command`` attachment (typed while the agent worked); '' otherwise."""
    attachment = record.attachment
    if record.type != 'attachment' or attachment is None or attachment.type != 'queued_command':
        return ''
    if not _is_human(attachment.origin):
        return ''
    prompt = attachment.prompt
    if isinstance(prompt, str):
        text = prompt
    elif prompt is None:
        return ''
    else:
        text = '\n'.join(block.text or '' for block in prompt if block.type == 'text')
    return _clean(text)


def fork_transcripts(session_dir: Path) -> Sequence[Path]:
    """Subagent/fork transcripts under a session directory (``agent-*.jsonl``, incl. ``subagents/``)."""
    paths: list[Path] = []
    for directory in (session_dir, session_dir / 'subagents'):
        if directory.is_dir():
            paths.extend(directory.glob('agent-*.jsonl'))
    return sorted(set(paths))


def parent_relayed(records: Sequence[TranscriptRecord]) -> Sequence[str]:
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


def is_relayed(text: str, relayed: Sequence[str]) -> bool:
    """True if ``text`` is (a prefix of) a directive the parent relayed into a fork."""
    normalized = _normalize(text)
    head = normalized[:MATCH_PREFIX]
    return any(head in candidate or candidate[:MATCH_PREFIX] in normalized for candidate in relayed)


def extract_spine(transcript_path: Path, *, include_forks: bool = True) -> Sequence[SpineEntry]:
    """The full user-intent spine for one session: normal turns + queued directives + fork-direct.

    Scoped to a single session by design — a predecessor generation is a *different* session and
    is never folded in here. Entries are chronological across sources; queued directives already
    seen as a normal turn (enqueued then processed) are de-duplicated by normalized text.
    """
    records = parse_transcript(transcript_path)
    rows: list[tuple[str, str, str]] = []  # (raw_iso_ts, text, source)
    seen: set[str] = set()

    for record in records:
        if text := human_text(record):
            rows.append((record.timestamp or '', text, 'user'))
            seen.add(_normalize(text))
    for record in records:
        if (text := queued_text(record)) and _normalize(text) not in seen:
            rows.append((record.timestamp or '', text, 'queued'))
            seen.add(_normalize(text))

    if include_forks:
        relayed = parent_relayed(records)
        session_dir = transcript_path.parent / transcript_path.stem
        for fork_path in fork_transcripts(session_dir):
            name = fork_path.name.removeprefix('agent-').removesuffix('.jsonl')
            rows.extend(
                (record.timestamp or '', text, f'fork:{name}')
                for record in parse_transcript(fork_path)
                if (text := human_text(record)) and not is_relayed(text, relayed)
            )

    rows.sort(key=lambda row: row[0])  # chronological across all sources (ISO-UTC strings sort correctly)
    return [SpineEntry(timestamp=_stamp(ts), text=text, source=source) for ts, text, source in rows]


def _is_human(origin: Origin | None) -> bool:
    return origin is None or origin.kind == 'human'


def _clean(text: str) -> str:
    text = text.strip()
    return '' if not text or text in SKIP_EXACT or text.startswith(SKIP_PREFIXES) else text


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def _stamp(timestamp: str) -> str:
    return f'{datetime.fromisoformat(timestamp).astimezone(PACIFIC):%Y-%m-%d %I:%M%p} PT' if timestamp else ''
