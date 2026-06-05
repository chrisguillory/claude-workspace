#!/usr/bin/env -S uv run
"""Repair operator-approved model-hallucination tool-args in session JSONL.

These are data-hallucinations, not schema gaps (binary-proven gate: the binary writes the
correct field/type; the model emitted a malformed tool arg). The fix is the data, not the
schema. Two patterns:

  - TaskUpdate input ``id`` -> ``taskId`` (the binary schema uses taskId).
  - Stringized dash-flags: ``-n``/``-i``/``-r``/``-o`` "true"/"false" -> bool;
    ``-A``/``-B``/``-C`` "<n>" -> int (the binary schema types these bool/int).

Two phases, so nothing is auto-edited:

  REPORT (default)  uv run --project mcp/claude-session scripts/clean_hallucinations.py
      Lists every fixable-failing record as ``<hash>  <file>:<line>  <tool>: <fix>``. The
      hash is the sha256 prefix of the exact record line.
  FIX               ... --fix <hash> [--fix <hash> ...]
      Repairs ONLY the approved hashes. Because the hash is content-addressed, a record
      whose line changed since the report no longer matches and is skipped (reported).

Safety: a record is a finding only if it currently FAILS ``validate_session_record`` AND
PASSES after the fix. Each modified file is backed up first. ``--exclude`` skips sessions
by id substring (always exclude the active session — never rewrite a file Claude Code is
appending to). Journal files (workflow run-journals, a different schema) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pydantic
from claude_session.schemas.session.models import validate_session_record

_BOOL_FLAGS = {'-n', '-i', '-r', '-o'}
_INT_FLAGS = {'-A', '-B', '-C'}


class Finding(NamedTuple):
    """One fixable-failing record, addressed by the content hash of its line."""

    hash: str
    path: Path
    lineno: int
    fixes: Sequence[str]
    fixed_line: str


def _short_hash(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()[:7]


def _fixed_input(name: str | None, inp: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Sequence[str]]:
    """Return (repaired copy, fix descriptions), or (None, []) if nothing is fixable."""
    out = dict(inp)
    fixes: list[str] = []
    if name == 'TaskUpdate' and 'id' in out and 'taskId' not in out:
        out['taskId'] = out.pop('id')
        fixes.append('id->taskId')
    for key in list(out):
        val = out[key]
        if key in _BOOL_FLAGS and isinstance(val, str) and val in ('true', 'false'):
            out[key] = val == 'true'
            fixes.append(f'{key}:"{val}"->bool')
        elif key in _INT_FLAGS and isinstance(val, str) and val.lstrip('-').isdigit():
            out[key] = int(val)
            fixes.append(f'{key}:"{val}"->int')
    return (out, fixes) if fixes else (None, [])


def _fixed_record(record: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Sequence[str]]:
    """Return (repaired record, fix descriptions), or (None, []). Pure -- input is untouched."""
    message = record.get('message')
    if not isinstance(message, dict):
        return None, []
    content = message.get('content')
    if not isinstance(content, list):
        return None, []
    new_content: list[Any] = []
    fixes: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_use' and isinstance(block.get('input'), dict):
            fixed, block_fixes = _fixed_input(block.get('name'), block['input'])
            if fixed is not None:
                block = {**block, 'input': fixed}
                fixes.extend(f'{block.get("name")}:{f}' for f in block_fixes)
        new_content.append(block)
    if not fixes:
        return None, []
    return {**record, 'message': {**message, 'content': new_content}}, fixes


def _validates(record: Mapping[str, Any]) -> bool:
    try:
        validate_session_record(record)
        return True
    except pydantic.ValidationError:
        return False


def findings(root: Path, excludes: Sequence[str]) -> Iterator[Finding]:
    """Yield a Finding for every record that fails validation but passes after the fix."""
    for path in sorted(root.glob('**/*.jsonl')):
        if path.name == 'journal.jsonl' or any(x in path.name for x in excludes):
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            fixed, fixes = _fixed_record(record)
            if fixed is None or _validates(record) or not _validates(fixed):
                continue
            yield Finding(_short_hash(stripped), path, lineno, fixes, json.dumps(fixed, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default=str(Path.home() / '.claude/projects'))
    parser.add_argument(
        '--exclude', action='append', default=[], help='session-id substring to skip (e.g. active session)'
    )
    parser.add_argument(
        '--fix', action='append', default=[], metavar='HASH', help='hash(es) to repair; omit to report only'
    )
    parser.add_argument('--backup', default=str(Path.home() / '.claude-workspace/claude-session/cleaned'))
    args = parser.parse_args()

    approved = set(args.fix)
    found = list(findings(Path(args.root), tuple(args.exclude)))

    if not approved:
        for f in found:
            print(f'{f.hash}  {f.path.name}:{f.lineno}  {", ".join(f.fixes)}')
        print(f'\n{len(found)} fixable finding(s). Re-run with --fix <hash> [...] to repair the approved ones.')
        return

    backup_dir = Path(args.backup) / datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    by_file: dict[Path, dict[int, str]] = {}
    for f in found:
        if f.hash in approved:
            by_file.setdefault(f.path, {})[f.lineno] = f.fixed_line

    fixed_total = 0
    for path, replacements in by_file.items():
        lines = path.read_text().splitlines()
        for lineno, fixed_line in replacements.items():
            lines[lineno - 1] = fixed_line
            fixed_total += 1
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
        path.write_text('\n'.join(lines) + '\n')
        print(f'  {path.name}: {len(replacements)} record(s)')

    missing = approved - {f.hash for f in found}
    if missing:
        print(
            f'WARNING: {len(missing)} approved hash(es) not found (file changed, or already fixed): {", ".join(sorted(missing))}'
        )
    print(f'fixed {fixed_total} record(s) across {len(by_file)} file(s); backups in {backup_dir}')


if __name__ == '__main__':
    main()
