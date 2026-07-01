"""Regression tests for transcript_spine: the origin-kind filter and the queued_command wire shape."""

from __future__ import annotations

import json
from pathlib import Path

from cc_lib.transcript_spine import TranscriptRecord, extract_spine, human_text, queued_text


def test_human_text_keeps_human_kind() -> None:
    """CC 2.1.x tags every user turn with origin; kind=='human' is a real directive, not relay.

    The stale `origin is not None -> drop` (still live in recover-session pre-migration) dropped these.
    """
    record = TranscriptRecord.model_validate(
        {'type': 'user', 'origin': {'kind': 'human'}, 'message': {'content': 'do X'}}
    )
    assert human_text(record) == 'do X'


def test_human_text_keeps_origin_absent() -> None:
    """No origin = older Claude Code, or a bridge-relayed human turn (Slack/Teams) — both kept."""
    record = TranscriptRecord.model_validate({'type': 'user', 'message': {'content': 'do Y'}})
    assert human_text(record) == 'do Y'


def test_human_text_drops_machine_origins() -> None:
    """The five machine origin kinds (binary-verified producers) are relays, not keystrokes."""
    for kind in ('task-notification', 'auto-continuation', 'channel', 'coordinator', 'peer'):
        record = TranscriptRecord.model_validate(
            {'type': 'user', 'origin': {'kind': kind}, 'message': {'content': 'relayed'}}
        )
        assert human_text(record) == '', kind


def test_human_text_drops_meta_and_scaffolding() -> None:
    meta = TranscriptRecord.model_validate({'type': 'user', 'isMeta': True, 'message': {'content': 'injected'}})
    wrapper = TranscriptRecord.model_validate({'type': 'user', 'message': {'content': '<task-notification>x'}})
    assert human_text(meta) == ''
    assert human_text(wrapper) == ''


def test_human_text_drops_cc_injected_fork_prompts() -> None:
    """CC-injected fork spawn prompts (compaction / probe forks) are scaffolding, not user directives.

    These reach a subagent transcript with origin=None and no `<` prefix, so only a content-prefix
    filter catches them (a fork-name exclusion would wrongly drop a real directive that landed in a
    compact-named fork).
    """
    compaction = TranscriptRecord.model_validate(
        {
            'type': 'user',
            'message': {'content': 'Your task is to create a detailed summary of the conversation so far.'},
        }
    )
    probe = TranscriptRecord.model_validate(
        {'type': 'user', 'message': {'content': 'CRITICAL: Respond with TEXT ONLY, no tool calls.'}}
    )
    assert human_text(compaction) == ''
    assert human_text(probe) == ''


def test_queued_text_string_prompt() -> None:
    record = TranscriptRecord.model_validate(
        {'type': 'attachment', 'attachment': {'type': 'queued_command', 'prompt': 'queued ask'}}
    )
    assert queued_text(record) == 'queued ask'


def test_queued_text_extracts_content_block_prompt() -> None:
    """A multimodal queued_command (prompt = content-block list) must parse and yield its text.

    Wire shape is `str | (TextContent|ImageContent)[]`; an earlier `Sequence[str]` typing made
    parse_transcript swallow the ValidationError and silently lose the directive.
    """
    record = TranscriptRecord.model_validate(
        {
            'type': 'attachment',
            'attachment': {
                'type': 'queued_command',
                'prompt': [
                    {'type': 'text', 'text': 'fix the PR — '},
                    {'type': 'image', 'source': {'media_type': 'image/png', 'data': '...'}},
                    {'type': 'text', 'text': 'it links bad issues'},
                ],
            },
        }
    )
    assert queued_text(record) == 'fix the PR — \nit links bad issues'


def test_extract_spine_captures_multimodal_queued(tmp_path: Path) -> None:
    """End-to-end: a multimodal queued directive survives extraction (dropped at parse pre-fix)."""
    transcript = tmp_path / 'session.jsonl'
    records = [
        {
            'type': 'user',
            'origin': {'kind': 'human'},
            'timestamp': '2026-06-19T20:00:00Z',
            'message': {'content': 'first directive'},
        },
        {
            'type': 'attachment',
            'timestamp': '2026-06-19T20:05:00Z',
            'attachment': {
                'type': 'queued_command',
                'prompt': [{'type': 'text', 'text': 'queued multimodal directive'}],
            },
        },
    ]
    transcript.write_text('\n'.join(json.dumps(record) for record in records))
    texts = [entry.text for entry in extract_spine(transcript, include_forks=False)]
    assert texts == ['first directive', 'queued multimodal directive']


def test_extract_spine_dedups_queued_already_seen_as_turn(tmp_path: Path) -> None:
    """A directive enqueued then processed as a normal turn appears once, not twice."""
    transcript = tmp_path / 'session.jsonl'
    records = [
        {
            'type': 'user',
            'origin': {'kind': 'human'},
            'timestamp': '2026-06-19T20:00:00Z',
            'message': {'content': 'same ask'},
        },
        {
            'type': 'attachment',
            'timestamp': '2026-06-19T20:01:00Z',
            'attachment': {'type': 'queued_command', 'prompt': 'same ask'},
        },
    ]
    transcript.write_text('\n'.join(json.dumps(record) for record in records))
    texts = [entry.text for entry in extract_spine(transcript, include_forks=False)]
    assert texts == ['same ask']


def test_human_text_drops_suggestion_and_warmup_forks() -> None:
    """CC autocomplete (SUGGESTION MODE) and cache-warmup (Warmup) forks are injected, not directives.

    Surfaced as the top fork-direct leaks (295 + 169) by the corpus audit. Warmup is matched exactly,
    not as a prefix, so a real 'Warmup the cache' main directive still survives.
    """
    suggestion = TranscriptRecord.model_validate(
        {'type': 'user', 'message': {'content': '[SUGGESTION MODE: Suggest what the user might type next...]'}}
    )
    warmup = TranscriptRecord.model_validate({'type': 'user', 'message': {'content': 'Warmup'}})
    real_directive = TranscriptRecord.model_validate(
        {'type': 'user', 'message': {'content': 'Warmup the cache please'}}
    )
    assert human_text(suggestion) == ''
    assert human_text(warmup) == ''
    assert human_text(real_directive) == 'Warmup the cache please'
