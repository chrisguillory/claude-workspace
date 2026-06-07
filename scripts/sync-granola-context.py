#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
#     "document-search",
#     "httpx",
#     "markdownify",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../cc-lib/", editable = true }
# document-search = { path = "../mcp/document-search/", editable = true }
# ///
"""Sync Granola meetings to local archive.

Called by the /sync-granola-context skill. Fetches the full meeting list from the
Granola API, saves to all_meetings.json, compares against completed_ids.txt,
and downloads missing notes + transcripts. Fully deterministic.

Usage:
    scripts/sync-granola-context.py   # archive at <repo-root>/context/granola (cwd-independent)
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence, Set
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import markdownify
from cc_lib import ErrorBoundary
from cc_lib.schemas.base import SubsetModel
from document_search.schemas.indexing import IndexingResult

# Force line-buffered output for progress visibility. typeshed types the live
# streams as TextIO (they are legally monkeypatchable), so narrow to the
# concrete io.TextIOWrapper — the only type declaring reconfigure() — per the
# isinstance guard the typeshed stub itself documents.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True)
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anchored to this script's location (scripts/ → parents[1] is the repo root) so
# resolution is cwd-independent and survives invocation by absolute path from
# other projects.
ARCHIVE_DIR = Path(__file__).resolve().parents[1] / 'context' / 'granola'
MEETINGS_JSON = ARCHIVE_DIR / 'all_meetings.json'
COMPLETED_IDS = ARCHIVE_DIR / 'completed_ids.txt'
MEETINGS_DIR = ARCHIVE_DIR / 'meetings'

GRANOLA_API = 'https://api.granola.ai'

# Granola stopped refreshing supabase.json after its March-2026 DB-encryption
# change, so the static access_token there is permanently expired; a valid token
# must be minted via /v1/refresh-access-token, and the API rejects requests
# lacking Electron identity headers with a 200 "Unsupported client" envelope.
# granola-mcp implements both correctly — reuse it as the single source of auth
# truth rather than duplicating (and drifting from) it.
GRANOLA_MCP_DIR = Path.home() / 'granola-mcp'


# ---------------------------------------------------------------------------
# Models (vocabulary — subset of Granola API responses)
# ---------------------------------------------------------------------------


class DocumentSetEntry(SubsetModel):
    """Lightweight entry from /v1/get-document-set."""

    updated_at: str | None = None
    owner: bool | None = None


class Meeting(SubsetModel):
    """Meeting metadata from /v1/get-documents-batch."""

    id: str
    title: str | None = None
    created_at: str | None = None
    deleted_at: str | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def get_auth_headers() -> Mapping[str, str]:
    """Authenticated + Electron identity headers, via granola-mcp's auth.

    Fails loudly if granola-mcp is unavailable rather than falling back to the
    stale static token: that fallback returns an empty document set, which the
    cleanup step previously misread as "every meeting was deleted" and wiped the
    local archive.
    """
    if str(GRANOLA_MCP_DIR) not in sys.path:
        sys.path.insert(0, str(GRANOLA_MCP_DIR))
    try:
        from granola_mcp.helpers import (  # noqa: PLC0415 — needs sys.path injection above
            get_auth_headers as _mcp_auth_headers,
        )
    except ImportError as exc:
        raise RuntimeError(f'Granola auth requires granola-mcp at {GRANOLA_MCP_DIR}. Import failed: {exc}') from exc
    return _mcp_auth_headers()


# ---------------------------------------------------------------------------
# Meeting list fetch (replaces MCP list_meetings + jq save step)
# ---------------------------------------------------------------------------


def _fetch_meeting_list(client: httpx.Client, headers: Mapping[str, str]) -> Sequence[Mapping[str, Any]]:
    """Fetch all meetings from Granola API.

    Two-step process matching the Granola MCP server:
    1. POST /v1/get-document-set {} → lightweight index of all doc IDs
    2. POST /v1/get-documents-batch {document_ids: [...]} → full details (chunks of 200)
    """
    # Step 1: lightweight index
    index_resp = client.post(f'{GRANOLA_API}/v1/get-document-set', json={}, headers=headers)
    index_resp.raise_for_status()
    doc_index = index_resp.json()

    # Extract owned document IDs
    doc_ids = [
        doc_id
        for doc_id, entry in doc_index.get('documents', {}).items()
        if DocumentSetEntry.model_validate(entry).owner is True
    ]
    print(f'Found {len(doc_ids)} owned documents in index', file=sys.stderr)

    # Step 2: batch fetch full details. api.granola.ai now caps get-documents-batch
    # below 100 ids/request (HTTP 400 otherwise); granola-mcp uses 40.
    BATCH_SIZE = 40
    all_meetings: list[dict[str, Any]] = []
    for i in range(0, len(doc_ids), BATCH_SIZE):
        chunk = doc_ids[i : i + BATCH_SIZE]
        batch_resp = client.post(
            f'{GRANOLA_API}/v1/get-documents-batch',
            json={'document_ids': chunk},
            headers=headers,
        )
        batch_resp.raise_for_status()
        docs = batch_resp.json().get('docs', [])
        all_meetings.extend(d for d in docs if not d.get('deleted_at'))

    all_meetings.sort(key=lambda m: m.get('created_at', ''), reverse=True)
    return all_meetings


# ---------------------------------------------------------------------------
# ProseMirror → Markdown conversion
# ---------------------------------------------------------------------------


def _prosemirror_to_markdown(content: Mapping[str, Any], depth: int = 0) -> str:
    """Convert ProseMirror JSON to Markdown."""
    if not isinstance(content, dict):
        return ''

    node_type = content.get('type', '')

    if node_type == 'doc':
        children = content.get('content', [])
        return '\n\n'.join(_prosemirror_to_markdown(child, depth) for child in children)

    if node_type == 'heading':
        level = content.get('attrs', {}).get('level', 1)
        return f'{"#" * level} {_extract_text(content)}'

    if node_type == 'paragraph':
        text = _extract_text(content)
        return text if text else ''

    if node_type == 'horizontalRule':
        return '---'

    if node_type == 'bulletList':
        lines: list[str] = []
        for item in content.get('content', []):
            if item.get('type') == 'listItem':
                lines.extend(_process_list_item(item, depth))
        return '\n'.join(lines)

    if node_type == 'orderedList':
        lines = []
        for i, item in enumerate(content.get('content', []), 1):
            if item.get('type') == 'listItem':
                lines.extend(_process_list_item(item, depth, ordered=i))
        return '\n'.join(lines)

    if node_type == 'codeBlock':
        return f'```\n{_extract_text(content)}\n```'

    return _extract_text(content)


# ---------------------------------------------------------------------------
# Per-meeting download
# ---------------------------------------------------------------------------


def download_note(client: httpx.Client, document_id: str, headers: Mapping[str, str]) -> str | None:
    """Download AI-generated meeting notes. Returns None if no panels exist."""
    document = _get_document(client, document_id, headers)

    panels_resp = client.post(
        f'{GRANOLA_API}/v1/get-document-panels',
        json={'document_id': document_id},
        headers=headers,
    )
    panels_resp.raise_for_status()
    panels_data = panels_resp.json()

    if not panels_data:
        return None

    # Prefer consolidated summary panel
    summary_panel = next(
        (p for p in panels_data if p.get('template_slug') == 'v2:meeting-summary-consolidated'),
        panels_data[0],
    )

    content = summary_panel.get('content', {})
    if isinstance(content, str):
        notes_md = markdownify.markdownify(
            content,
            heading_style='ATX',
            bullets='-',
            default_title=True,
        ).strip()
    else:
        notes_md = _prosemirror_to_markdown(content)

    date_str = _format_date(document.get('created_at', ''), '%a, %d %b %y')
    title = document.get('title') or '(Untitled)'
    return f'# {title}\n\n{date_str}\n\n{notes_md}'


def download_transcript(client: httpx.Client, document_id: str, headers: Mapping[str, str]) -> str | None:
    """Download meeting transcript. Returns None if transcript doesn't exist."""
    document = _get_document(client, document_id, headers)

    resp = client.post(
        f'{GRANOLA_API}/v1/get-document-transcript',
        json={'document_id': document_id},
        headers=headers,
    )
    resp.raise_for_status()
    segments = resp.json()

    if not segments:
        return None

    title = document.get('title') or '(Untitled)'
    date_str = _format_date(document.get('created_at', ''), '%b %-d')

    lines = [f'Meeting Title: {title}', f'Date: {date_str}', '', 'Transcript:', ' ']

    # Combine consecutive segments from same speaker
    current_label: str | None = None
    current_texts: list[str] = []

    for segment in segments:
        label = 'Me' if segment.get('source') == 'microphone' else 'Them'
        if label == current_label:
            current_texts.append(segment.get('text', ''))
        else:
            if current_label is not None:
                lines.append(f'{current_label}: {" ".join(current_texts)}  ')
            current_label = label
            current_texts = [segment.get('text', '')]

    if current_label is not None:
        lines.append(f'{current_label}: {" ".join(current_texts)} ')

    return '\n'.join(lines)


def download_private_notes(client: httpx.Client, document_id: str, headers: Mapping[str, str]) -> str | None:
    """Download private notes. Returns None if no private notes exist."""
    document = _get_document(client, document_id, headers)

    if not document.get('notes_markdown'):
        return None

    date_str = _format_date(document.get('created_at', ''), '%a, %d %b %y')
    title = document.get('title') or '(Untitled)'
    return f'# {title}\n\n{date_str}\n\n{document["notes_markdown"]}'


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def cleanup_deleted(api_ids: Set[str], completed: Set[str]) -> Sequence[str]:
    """Remove local files and tracking for meetings deleted from Granola.

    Returns list of removed meeting IDs.
    """
    # Defense-in-depth: empty api_ids means the fetch degenerated (auth or
    # identity-header failure behind an HTTP 200), not that every meeting was
    # deleted. Never treat that as a cleanup signal.
    if not api_ids:
        return []

    stale_ids = completed - api_ids
    if not stale_ids:
        return []

    removed: list[str] = []
    for mid in stale_ids:
        # Remove local files
        for suffix in ('note', 'transcript', 'private'):
            path = MEETINGS_DIR / f'{mid}-{suffix}.md'
            if path.exists():
                path.unlink()
        removed.append(mid)
        print(f'  Cleaned up: {mid}', file=sys.stderr)

    # Rewrite completed_ids.txt without the stale IDs
    if removed:
        remaining = completed - stale_ids
        COMPLETED_IDS.write_text('\n'.join(sorted(remaining)) + '\n', encoding='utf-8')

    return removed


def index_meetings(meetings_dir: Path) -> IndexingResult:
    """Index the meetings directory via document-search CLI."""
    print('Indexing meetings for semantic search...', file=sys.stderr)
    result = subprocess.run(
        ['document-search', 'index', str(meetings_dir), '-c', 'document-chunks', '--no-gitignore', '--format', 'json'],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    index_result = IndexingResult.model_validate(json.loads(result.stdout))
    print(
        f'Indexed: {index_result.files_indexed} files, {index_result.chunks_created} new chunks'
        f' ({index_result.files_cached} cached)',
        file=sys.stderr,
    )
    return index_result


def format_output(
    total: int,
    previously_synced: int,
    downloaded: Sequence[tuple[str, str]],
    skipped_notes: Sequence[tuple[str, str]],
    index_result: IndexingResult,
    cleaned_up: int = 0,
) -> str:
    """Format sync results as structured text for model consumption."""
    lines = [
        '## Granola Sync Results\n',
        f'Total meetings: {total}',
        f'Previously synced: {previously_synced}',
        f'Downloaded: {len(downloaded)}',
    ]

    if cleaned_up:
        lines.append(f'Cleaned up (deleted from Granola): {cleaned_up}')

    lines.append(
        f'Index: {index_result.files_indexed} new, {index_result.files_cached} cached,'
        f' {index_result.chunks_created} chunks created'
    )

    lines.append('')

    if not downloaded and not skipped_notes and not cleaned_up:
        lines.append('All meetings already synced.')
        return '\n'.join(lines)

    if downloaded:
        lines.append(f'### Downloaded ({len(downloaded)})\n')
        for title, date in downloaded:
            lines.append(f'- {title} ({date[:10] if date else "unknown"})')

    if skipped_notes:
        lines.append(f'\n### No notes available ({len(skipped_notes)})\n')
        for title, date in skipped_notes:
            lines.append(f'- {title} ({date[:10] if date else "unknown"})')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_document(client: httpx.Client, document_id: str, headers: Mapping[str, str]) -> Mapping[str, Any]:
    """Fetch a single document's metadata."""
    resp = client.post(f'{GRANOLA_API}/v2/get-documents', json={'id': document_id}, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    if not data.get('docs'):
        raise ValueError(f'Document {document_id} not found')
    document: Mapping[str, Any] = data['docs'][0]
    return document


def _format_date(iso_timestamp: str, fmt: str) -> str:
    """Convert ISO timestamp to formatted local date string."""
    if not iso_timestamp:
        return ''
    dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
    return dt.astimezone().strftime(fmt)


def _extract_text(node: object) -> str:
    """Recursively extract all text from a ProseMirror node."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ''

    if node.get('type') == 'text':
        text: str = node.get('text', '')
        for mark in node.get('marks', []):
            mark_type = mark.get('type')
            if mark_type == 'bold':
                text = f'**{text}**'
            elif mark_type == 'italic':
                text = f'*{text}*'
            elif mark_type == 'code':
                text = f'`{text}`'
            elif mark_type == 'link':
                href = mark.get('attrs', {}).get('href', '')
                if href:
                    text = f'[{text}]({href})'
        return text

    texts = [_extract_text(child) for child in node.get('content', [])]
    if node.get('type') in ('paragraph', 'listItem'):
        return re.sub(r' +', ' ', ' '.join(t for t in texts if t))
    return ''.join(texts)


def _process_list_item(item: Mapping[str, Any], depth: int, ordered: int | None = None) -> Sequence[str]:
    """Process a list item with support for nested lists."""
    indent = '  ' * depth
    bullet = f'{ordered}.' if ordered else '-'
    first_line_parts: list[str] = []
    nested_content: list[str] = []

    for node in item.get('content', []):
        node_type = node.get('type', '')
        if node_type == 'paragraph':
            text = _extract_text(node)
            if text:
                first_line_parts.append(text)
        elif node_type in ('bulletList', 'orderedList'):
            nested_md = _prosemirror_to_markdown(node, depth + 1)
            if nested_md:
                nested_content.append(nested_md)

    return [f'{indent}{bullet} {" ".join(first_line_parts)}', *nested_content]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@ErrorBoundary(exit_code=1)
def main() -> int:
    """Fetch meeting list, diff against local state, download missing."""
    MEETINGS_DIR.mkdir(exist_ok=True)
    headers = get_auth_headers()

    with httpx.Client(timeout=30.0) as client:
        # Phase 1: Fetch + save meeting list
        print('Fetching meeting list from Granola API...', file=sys.stderr)
        all_meetings = _fetch_meeting_list(client, headers)

        # Safety guard: a degenerate fetch (auth/identity-header failure returns an
        # empty or truncated set behind an HTTP 200) must never clobber the saved
        # list or feed the cleanup step. Bail out before any destructive write.
        prior_count = 0
        if COMPLETED_IDS.exists():
            prior_count = sum(1 for ln in COMPLETED_IDS.read_text().splitlines() if ln.strip())
        if not all_meetings:
            print(
                'ERROR: Granola returned 0 meetings — almost certainly an auth or '
                'identity-header failure, not an empty account. Refusing to overwrite '
                'all_meetings.json or clean up local files. Aborting with no changes.',
                file=sys.stderr,
            )
            return 1
        if prior_count and len(all_meetings) < prior_count // 2:
            print(
                f'ERROR: Granola returned {len(all_meetings)} meetings but the archive '
                f'has {prior_count}. A drop this large signals a fetch failure, not real '
                f'deletions. Refusing to proceed. Aborting with no changes.',
                file=sys.stderr,
            )
            return 1

        MEETINGS_JSON.write_text(json.dumps(all_meetings, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Saved {len(all_meetings)} meetings to all_meetings.json', file=sys.stderr)

        # Phase 2: Diff against completed
        completed: set[str] = set()
        if COMPLETED_IDS.exists():
            completed = {line.strip() for line in COMPLETED_IDS.read_text().splitlines() if line.strip()}

        to_process = [m for m in all_meetings if m['id'] not in completed]

        # Phase 2b: Clean up meetings deleted from Granola
        api_ids = {m['id'] for m in all_meetings}
        removed = cleanup_deleted(api_ids, completed)
        if removed:
            print(f'Cleaned up {len(removed)} deleted meetings', file=sys.stderr)
            # Update completed count after cleanup
            completed -= set(removed)

        # Always index — incremental, skips cached files
        if not to_process:
            idx = index_meetings(MEETINGS_DIR)
            print(format_output(len(all_meetings), len(completed), [], [], idx, len(removed)))
            return 0

        # Phase 3: Download missing
        downloaded: list[tuple[str, str]] = []
        skipped_notes: list[tuple[str, str]] = []

        for i, meeting in enumerate(to_process, 1):
            mid = meeting['id']
            title = meeting.get('title') or '(Untitled)'
            created = meeting.get('created_at', '')
            print(f'[{i}/{len(to_process)}] {title[:60]}', file=sys.stderr)

            note = download_note(client, mid, headers)
            if note:
                (MEETINGS_DIR / f'{mid}-note.md').write_text(note, encoding='utf-8')
                print(f'  Note: {len(note)} bytes', file=sys.stderr)
            else:
                skipped_notes.append((title, created))
                print('  No note available', file=sys.stderr)

            transcript = download_transcript(client, mid, headers)
            if transcript:
                (MEETINGS_DIR / f'{mid}-transcript.md').write_text(transcript, encoding='utf-8')
                print(f'  Transcript: {len(transcript)} bytes', file=sys.stderr)

            private = download_private_notes(client, mid, headers)
            if private:
                (MEETINGS_DIR / f'{mid}-private.md').write_text(private, encoding='utf-8')
                print(f'  Private notes: {len(private)} bytes', file=sys.stderr)

            with open(COMPLETED_IDS, 'a') as f:
                f.write(f'{mid}\n')

            downloaded.append((title, created))

            if i < len(to_process):
                time.sleep(5)

        # Phase 4: Index for semantic search
        idx = index_meetings(MEETINGS_DIR)

        # Phase 5: Report
        print(format_output(len(all_meetings), len(completed), downloaded, skipped_notes, idx, len(removed)))

    return 0


if __name__ == '__main__':
    sys.exit(main())
