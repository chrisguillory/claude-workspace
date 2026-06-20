#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../", editable = true }
# ///
"""Audit cc_lib.transcript_spine against a machine's full Claude Code session corpus.

Runs the real extractor over ~/.claude/projects and reports the signals that go stale when Claude
Code's transcript schema drifts, so a rerun after each CC release catches the drift before a spine
silently loses (or invents) a directive:

- origin-kind universe — an `unknown_origin_kinds` entry is a new producer the 'human' filter
  (cc_lib.transcript_spine._is_human) hasn't learned; it will misclassify those turns until taught.
- attachment + queued-prompt wire shapes — a new attachment type or a queued prompt shape other than
  str/list is a parse-model gap.
- fork-direct leak candidates — a high-count identical prefix is a CC-injected spawn prompt the skip
  set (SKIP_PREFIXES / SKIP_EXACT) does not yet catch, leaking into spines as a fake directive.

Run cross-mesh (`crb mac-others "cd ~/claude-workspace && cc-lib/scripts/spine_corpus_audit.py"`)
to widen the corpus past one machine's history.

Usage: spine_corpus_audit.py [PROJECTS_DIR]   (defaults to ~/.claude/projects)
"""

from __future__ import annotations

import collections
import json
import os
import sys
from collections.abc import Set
from pathlib import Path

from cc_lib.transcript_spine import (
    fork_transcripts,
    human_text,
    is_relayed,
    parent_relayed,
    parse_transcript,
)

# The origin kinds the spine filter already accounts for (None + cc_lib.transcript_spine machine kinds).
# Anything outside this set is drift: a new CC producer the 'human' allow-list may need to learn.
KNOWN_ORIGIN_KINDS: Set[str | None] = {
    None,
    'human',
    'auto-continuation',
    'channel',
    'coordinator',
    'peer',
    'task-notification',
}


def _queued_shape(prompt: object) -> str:
    """Wire shape of a queued_command prompt — str, multimodal content-block list, or absent."""
    if isinstance(prompt, str):
        return 'str'
    if prompt is None:
        return 'none'
    return 'list'


def main() -> None:
    projects = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/.claude/projects'))
    main_files = fork_files = main_user = 0
    origin_kinds: collections.Counter[str | None] = collections.Counter()
    attachment_types: collections.Counter[str | None] = collections.Counter()
    queued_shapes: collections.Counter[str] = collections.Counter()
    fork_candidates: collections.Counter[str] = collections.Counter()

    # Main transcripts are <project>/<session>.jsonl; their forks live in the sibling <session>/ dir.
    # Pairing them lets is_relayed use the parent's own relayed set, exactly as extract_spine does.
    for main_path in sorted(projects.glob('*/*.jsonl')):
        main_files += 1
        records = parse_transcript(main_path)
        for record in records:
            if record.type == 'user' and not record.isMeta:
                main_user += 1
                origin_kinds[record.origin.kind if record.origin else None] += 1
            elif record.type == 'attachment' and record.attachment:
                attachment_types[record.attachment.type] += 1
                if record.attachment.type == 'queued_command':
                    queued_shapes[_queued_shape(record.attachment.prompt)] += 1

        relayed = parent_relayed(records)
        for fork_path in fork_transcripts(main_path.parent / main_path.stem):
            fork_files += 1
            for record in parse_transcript(fork_path):
                if (text := human_text(record)) and not is_relayed(text, relayed):
                    fork_candidates[' '.join(text.split())[:80]] += 1

    report = {
        'host': os.uname().nodename.split('.')[0],
        'main_files': main_files,
        'fork_files': fork_files,
        'main_user_records': main_user,
        'origin_kinds': {str(kind): count for kind, count in origin_kinds.most_common()},
        'unknown_origin_kinds': sorted(str(k) for k in origin_kinds if k not in KNOWN_ORIGIN_KINDS),
        'attachment_types': {str(kind): count for kind, count in attachment_types.most_common()},
        'queued_shapes': dict(queued_shapes),
        'fork_direct_candidates_top': dict(fork_candidates.most_common(12)),
    }
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
