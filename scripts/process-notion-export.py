#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# ///
"""Lay a Notion export zip into context/notion/{workspace}/ + index.json.

Unwraps the double-nested export (outer zip -> Part-N inner zip), strips the
single `Export-{id}/` wrapper, and streams each file to a sanitized path under
the workspace dir -- preserving Notion's hierarchy (for path-scoped search)
while capping path components below the macOS filename limit. Emits index.json
mapping each page id to its title, on-disk relpath, and notion.so URL.

Usage:
    process-notion-export.py <export.zip> [workspace-slug]   # slug defaults to "workspace"
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTION_DIR = REPO_ROOT / 'context' / 'notion'
MAX_COMPONENT = 180  # stay under the macOS 255-byte filename limit, with headroom
_ID = re.compile(r'[0-9a-f]{32}')  # Notion page id (dashless), trails most filenames


class PageEntry(NamedTuple):
    """One archived page: Notion id -> title, on-disk relpath, source URL."""

    id: str
    title: str
    path: str
    url: str


def _extract_parts(outer: Path, staging: Path) -> Sequence[Path]:
    """Extract the outer zip and return its inner Part-N zip(s)."""
    with zipfile.ZipFile(outer) as z:
        z.extractall(staging)
    return sorted(staging.rglob('*-Part-*.zip')) or sorted(staging.rglob('*.zip'))


def _common_prefix(names: Sequence[str]) -> str:
    """The single top-level dir to strip (the `Export-{id}/` wrapper), or ''."""
    tops = {n.split('/', 1)[0] for n in names if n and not n.startswith('/')}
    return f'{tops.pop()}/' if len(tops) == 1 else ''


def _sanitize(component: str) -> str:
    """Truncate an over-long path component, preserving its page id and extension."""
    if len(component) <= MAX_COMPONENT:
        return component
    stem, dot, ext = component.rpartition('.')
    if not dot:
        stem, ext = component, ''
    matches = list(_ID.finditer(stem))
    tail = matches[-1].group(0) if matches else ''
    head = stem[: max(MAX_COMPONENT - len(ext) - len(tail) - 2, 8)]
    return f'{head}-{tail}.{ext}' if dot else f'{head}-{tail}'


def _sanitize_relpath(relpath: str) -> str:
    return '/'.join(_sanitize(part) for part in relpath.split('/') if part)


def _title_and_id(stem: str) -> tuple[str, str]:
    """Split a markdown filename stem into (title, page id)."""
    matches = list(_ID.finditer(stem))
    if not matches:
        return stem.strip() or '(untitled)', ''
    last = matches[-1]
    return stem[: last.start()].strip() or '(untitled)', last.group(0)


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: process-notion-export.py <export.zip> [workspace-slug]', file=sys.stderr)
        return 2
    outer = Path(sys.argv[1]).expanduser()
    slug = sys.argv[2] if len(sys.argv) > 2 else 'workspace'
    target = NOTION_DIR / slug

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    entries: list[PageEntry] = []
    counts: Counter[str] = Counter()
    total_bytes = 0

    with tempfile.TemporaryDirectory(prefix='notion-stage-') as tmp:
        for part in _extract_parts(outer, Path(tmp)):
            with zipfile.ZipFile(part) as pz:
                prefix = _common_prefix(pz.namelist())
                for info in pz.infolist():
                    if info.is_dir():
                        continue
                    rel = info.filename[len(prefix) :] if info.filename.startswith(prefix) else info.filename
                    if not rel:
                        continue
                    dest = target / _sanitize_relpath(rel)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with pz.open(info) as src, dest.open('wb') as out:
                        shutil.copyfileobj(src, out)
                    ext = dest.suffix.lower().lstrip('.') or '(none)'
                    counts[ext] += 1
                    total_bytes += info.file_size
                    if ext == 'md':
                        title, page_id = _title_and_id(dest.stem)
                        if page_id:
                            entries.append(
                                PageEntry(page_id, title, _sanitize_relpath(rel), f'https://www.notion.so/{page_id}')
                            )

    (target / 'index.json').write_text(
        json.dumps([e._asdict() for e in entries], indent=2, ensure_ascii=False), encoding='utf-8'
    )

    print(f'workspace: {slug}  ->  {target}')
    print(f'pages indexed: {len(entries)}   files: {sum(counts.values())}   bytes: {total_bytes / 1e9:.2f} GB')
    print('by type: ' + ', '.join(f'{ext}={n}' for ext, n in counts.most_common(10)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
