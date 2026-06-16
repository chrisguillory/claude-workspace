#!/usr/bin/env -S uv run --quiet --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc_lib",
#     "pyyaml",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../../../cc-lib/", editable = true }
# ///
"""Structural conformance validator for a where-am-i quest-map — checks shape, not content.

The frontmatter is a `ClosedModel`; the body cross-checks (counts vs the tree, numbering, header,
arrows, footer) are the rest. The post-run gate, and the CI check against the committed example.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from cc_lib.schemas.base import ClosedModel
from pydantic import Field, ValidationError, model_validator

ROOT = re.compile(r'^\[(\d+)]\s+(✓\s+)?\S')  # a root line: "[N] ✓ title …" or "[N] title …"
HEADER = re.compile(r'^WHERE AM I — session ', re.MULTILINE)
FOOTER = '— open parent quests never popped back up to —'
UUID = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: validate-map.py <top.md>')
    issues = validate(Path(sys.argv[1]))
    if issues:
        print(f'INVALID — {len(issues)} issue(s):')
        for issue in issues:
            print(f'  ✗ {issue}')
        sys.exit(1)
    print('valid ✓')


def validate(path: Path) -> Sequence[str]:
    text = path.read_text()
    if not text.startswith('---\n'):
        return ['no YAML frontmatter']
    _, frontmatter, body = text.split('---\n', 2)

    issues: list[str] = []
    matches = (ROOT.match(line) for line in body.splitlines())
    root_lines = [m for m in matches if m is not None]
    numbers = [int(m.group(1)) for m in root_lines]
    marked = sum(1 for m in root_lines if m.group(2))
    if numbers != list(range(1, len(root_lines) + 1)):
        issues.append(f'root numbers must be sequential 1..N, got {numbers}')
    if not HEADER.search(body):
        issues.append('missing "WHERE AM I — session …" header line')
    if '->' in body:
        issues.append('ASCII "->" found in body; arrows must be →')
    if FOOTER not in body:
        issues.append(f'missing dangling-frame footer: {FOOTER!r}')

    try:
        meta = QuestMapMeta.model_validate(yaml.safe_load(frontmatter))
    except ValidationError as exc:
        issues += [f'frontmatter {".".join(map(str, e["loc"]))}: {e["msg"]}' for e in exc.errors()]
    else:
        if meta.roots.total != len(root_lines):
            issues.append(f'roots.total ({meta.roots.total}) != numbered roots in tree ({len(root_lines)})')
        if meta.roots.landed != marked:
            issues.append(f'roots.landed ({meta.roots.landed}) != ✓-marked roots ({marked})')
    return issues


class Session(ClosedModel):
    id: str = Field(pattern=UUID)
    title: str
    machine: str


class Span(ClosedModel):
    from_: date = Field(alias='from')
    to: date
    weeks: int


class Volume(ClosedModel):
    messages: int
    compactions: int
    subagents: int


class Roots(ClosedModel):
    total: int
    landed: int
    open: int

    @model_validator(mode='after')
    def _counts_reconcile(self) -> Roots:
        if self.landed + self.open != self.total:
            raise ValueError(f'landed ({self.landed}) + open ({self.open}) != total ({self.total})')
        return self


class QuestMapMeta(ClosedModel):
    artifact: Literal['where-am-i']
    schema_version: int = Field(alias='schema')
    session: Session
    span: Span
    volume: Volume
    skills: Mapping[str, int]
    roots: Roots
    top_mcp: Mapping[str, int] = Field(default_factory=dict)
    provenance: Sequence[Mapping[str, object]] = ()


if __name__ == '__main__':
    main()
