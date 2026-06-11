#!/usr/bin/env -S uv run
"""Repair operator-approved model-hallucination tool-args in session JSONL.

These are data-hallucinations, not schema gaps (binary-proven gate: the binary writes the
correct field/type; the model emitted a malformed tool arg). The fix is the data, not the
schema. Two repair kinds:

  - Type coercion (schema-driven): a stringized value is coerced to the type its tool model
    declares -- e.g. Grep ``-n``/``head_limit``, TaskOutput ``block``. No param list; bool/int
    fields are read from each ``{Tool}ToolInput`` model, so a new one needs no change here.
  - Field rename (the one mismatch the schema can't derive): TaskUpdate ``id`` -> ``taskId``.

Two phases, so nothing is auto-edited:

  REPORT (default)  uv run --project mcp/claude-session scripts/clean_hallucinations.py
      Lists every fixable-failing record as ``<hash>  <file>:<line>  <fixes>``. The hash is
      the sha256 prefix of the exact record line.
  FIX               ... --fix <hash> [--fix <hash> ...]
      Repairs ONLY the approved hashes. Because the hash is content-addressed, a record
      whose line changed since the report no longer matches and is skipped (reported).

Safety: a record is a finding only if it currently FAILS ``validate_session_record`` AND
PASSES after the fix. Each modified file is backed up first. ``--exclude`` skips sessions
by id substring (always exclude the active session -- never rewrite a file Claude Code is
appending to). Journal files (workflow run-journals, a different schema) are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence, Set
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import pydantic
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import ClosedModel
from claude_session.schemas.session import models as cc_models
from claude_session.schemas.session.models import validate_session_record

type JsonRecord = Mapping[str, Any]  # a raw JSON-parsed record or tool input

error_boundary = ErrorBoundary(exit_code=1)

# The one repair the schema can't derive: a field NAME the model got wrong (not a type).
RENAMES: Mapping[str, Mapping[str, str]] = {'TaskUpdate': {'id': 'taskId'}}


class Finding(ClosedModel):
    """One fixable-failing record, addressed by the content hash of its line."""

    hash: str
    path: Path
    lineno: int
    fixes: Sequence[str]
    fixed_line: str


class Cleaner:
    """Schema-driven, operator-approved repair of hallucinated tool-args in session JSONL.

    A run is configured with a corpus root, session-id excludes, and a backup root.
    ``scan`` discovers fixable-failing records; ``repair`` rewrites approved hashes. The
    repair logic itself is stateless (class/static methods): it derives bool/int param types
    from each tool model, with ``RENAMES`` the one field-name mismatch the schema can't derive.
    """

    def __init__(self, root: Path, excludes: Sequence[str], backup_root: Path) -> None:
        self.root = root
        self.excludes = excludes
        self.backup_root = backup_root

    def scan(self) -> tuple[Sequence[Finding], int]:
        """Return (fixable findings, count of failing records the cleaner cannot repair).

        A record is a fixable finding only if it currently fails ``validate_session_record``
        and passes after the fix. Records the fix touches but that still fail (a different
        malformed field) are counted as ``residual`` and surfaced, not silently dropped, so
        the operator never gets a false all-clear. Lines are split on newline only, matching
        the real validator's read so reported line numbers agree with it.
        """
        findings: list[Finding] = []
        residual = 0
        for path in sorted(self.root.glob('**/*.jsonl')):
            if path.name == 'journal.jsonl' or any(x in path.name for x in self.excludes):
                continue
            try:
                lines = path.read_text().split('\n')
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
                fixed, fixes = self.fixed_record(record)
                if fixed is None or self.validates(record):
                    continue
                if self.validates(fixed):
                    findings.append(
                        Finding(
                            hash=self.short_hash(stripped),
                            path=path,
                            lineno=lineno,
                            fixes=fixes,
                            fixed_line=json.dumps(fixed, ensure_ascii=False),
                        )
                    )
                else:
                    residual += 1
        return findings, residual

    def repair(self, found: Sequence[Finding], approved: Set[str]) -> None:
        """Rewrite only the approved hashes in place, backing up each touched file first.

        Re-reads each file and re-confirms the target line still hashes to the approved hash
        before writing (content-addressed end-to-end); a line that changed since the scan is
        skipped and reported, never blindly overwritten. Lines are split/joined on newline
        only, so untouched records keep their exact bytes.
        """
        backup_dir = self.backup_root / datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        by_file: dict[Path, list[Finding]] = {}
        for finding in found:
            if finding.hash in approved:
                by_file.setdefault(finding.path, []).append(finding)

        fixed_total = 0
        stale = 0
        for path, file_findings in by_file.items():
            lines = path.read_text().split('\n')
            applied = 0
            for finding in file_findings:
                idx = finding.lineno - 1
                if idx >= len(lines) or self.short_hash(lines[idx].strip()) != finding.hash:
                    stale += 1
                    continue
                lines[idx] = finding.fixed_line
                applied += 1
            if applied:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup_dir / path.name)
                path.write_text('\n'.join(lines))
                fixed_total += applied
                print(f'  {path.name}: {applied} record(s)')

        missing = approved - {finding.hash for finding in found}
        if missing:
            print(f'WARNING: {len(missing)} approved hash(es) not in the scan: {", ".join(sorted(missing))}')
        if stale:
            print(f'WARNING: {stale} record(s) changed since the scan; skipped (re-run report).')
        print(f'fixed {fixed_total} record(s) across {len(by_file)} file(s); backups in {backup_dir}')

    @staticmethod
    def report(found: Sequence[Finding], residual: int) -> None:
        """Print each fixable finding, then the count of failing-but-unfixable records."""
        for finding in found:
            print(f'{finding.hash}  {finding.path.name}:{finding.lineno}  {", ".join(finding.fixes)}')
        print(f'\n{len(found)} fixable finding(s). Re-run with --fix <hash> [...] to repair the approved ones.')
        if residual:
            print(
                f'{residual} record(s) fail validation but are not auto-fixable (a different malformed '
                'field) -- inspect with validate_models.py --errors.'
            )

    @classmethod
    def fixed_record(cls, record: JsonRecord) -> tuple[JsonRecord | None, Sequence[str]]:
        """Return (repaired record, fix descriptions), or (None, []). Pure -- input untouched."""
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
                fixed, block_fixes = cls.fixed_input(block.get('name'), block['input'])
                if fixed is not None:
                    block = {**block, 'input': fixed}
                    fixes.extend(f'{block.get("name")}:{f}' for f in block_fixes)
            new_content.append(block)
        if not fixes:
            return None, []
        return {**record, 'message': {**message, 'content': new_content}}, fixes

    @classmethod
    def fixed_input(cls, name: str | None, inp: JsonRecord) -> tuple[JsonRecord | None, Sequence[str]]:
        """Return (repaired copy, fix descriptions), or (None, []) if nothing is fixable.

        Type coercions are schema-driven -- a stringized value is coerced to the type its tool
        model declares -- so a new bool/int param needs no change here. The only hard-coded
        repair is the id->taskId field RENAME, a name mismatch the schema cannot derive.
        """
        out = dict(inp)
        fixes: list[str] = []
        for bad, good in RENAMES.get(name or '', {}).items():
            if bad in out and good not in out:
                out[good] = out.pop(bad)
                fixes.append(f'{bad}->{good}')
        typed = cls.typed_params(name)
        for key in list(out):
            val = out[key]
            if not isinstance(val, str):
                continue
            expected = typed.get(key)
            if expected is bool and val in ('true', 'false'):
                out[key] = val == 'true'
                fixes.append(f'{key}:"{val}"->bool')
            elif expected is int and re.fullmatch(r'-?\d+', val, re.ASCII):
                out[key] = int(val)
                fixes.append(f'{key}:"{val}"->int')
        return (out, fixes) if fixes else (None, [])

    @classmethod
    def typed_params(cls, name: str | None) -> Mapping[str, type]:
        """Map a tool's JSON input keys to ``bool``/``int``, read from its ``{Tool}ToolInput`` model.

        Schema-driven: the bool/int fields come from the typed model itself, so a newly stringized
        param is coerced with no change here -- the structure is the source of truth, not a list.
        """
        fields = getattr(getattr(cc_models, f'{name}ToolInput', None), 'model_fields', None)
        if not isinstance(fields, dict):
            return {}
        typed: dict[str, type] = {}
        for fname, finfo in fields.items():
            base = cls.scalar_type(finfo.annotation)
            if base is not None:
                typed[finfo.alias or fname] = base
        return typed

    @staticmethod
    def scalar_type(
        annotation: Any,
    ) -> type | None:  # strict_typing_linter.py: loose-typing -- a runtime type annotation
        """If an annotation is purely ``bool`` or ``int`` (modulo ``None``), return it, else None."""
        args = [a for a in get_args(annotation) if a is not type(None)]
        base = args[0] if len(args) == 1 else (annotation if not args else None)
        return base if base in (bool, int) else None

    @staticmethod
    def validates(record: JsonRecord) -> bool:
        """Whether a record passes strict ``validate_session_record`` (the operations' real gate)."""
        try:
            validate_session_record(record)
            return True
        except pydantic.ValidationError:
            return False

    @staticmethod
    def short_hash(line: str) -> str:
        """The 7-char sha256 prefix that content-addresses a record line."""
        return hashlib.sha256(line.encode()).hexdigest()[:7]


@error_boundary
def main() -> None:
    """Report fixable hallucination records, or ``--fix`` the operator-approved hashes."""
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

    cleaner = Cleaner(Path(args.root), tuple(args.exclude), Path(args.backup))
    found, residual = cleaner.scan()
    approved = set(args.fix)
    if approved:
        cleaner.repair(found, approved)
    else:
        cleaner.report(found, residual)


if __name__ == '__main__':
    main()
