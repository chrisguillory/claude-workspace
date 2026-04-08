from __future__ import annotations

__all__ = [
    'ARIA_HIDDEN_REASON_KEYS',
    'VISUAL_HIDDEN_REASON_KEYS',
    'build_page_metadata',
    'build_storage_init_script',
    'count_tree_nodes',
    'save_large_output_to_file',
]

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ProfileState

ARIA_HIDDEN_REASON_KEYS: Sequence[tuple[str, str]] = [
    ('ariaHidden', 'aria-hidden'),
    ('displayNone', 'display-none'),
    ('visibilityHidden', 'visibility-hidden'),
    ('inert', 'inert'),
    ('other', 'other'),
]

VISUAL_HIDDEN_REASON_KEYS: Sequence[tuple[str, str]] = [
    ('displayNone', 'display-none'),
    ('visibilityHidden', 'visibility-hidden'),
    ('opacity', 'opacity'),
    ('clipped', 'clipped'),
    ('offscreen', 'offscreen'),
    ('other', 'other'),
]


def count_tree_nodes(
    node: Mapping[str, Any] | None,
) -> int:  # strict_typing_linter.py: loose-typing — tree nodes are untyped JS objects
    """Count nodes in a tree for compaction stats."""
    if node is None:
        return 0
    count = 1
    for child in node.get('children', []):
        count += count_tree_nodes(child)
    return count


def build_page_metadata(
    page_stats: Mapping[str, Any],  # strict_typing_linter.py: loose-typing — JS page stats are untyped
    include_page_info: bool,
    include_urls: bool,
    compact_tree: bool,
    raw_node_count: int,
    compacted_node_count: int,
    hidden_reason_keys: Sequence[tuple[str, str]],
) -> str:
    """Build YAML comment metadata footer from JS page stats.

    Tier 1 (always shown when > 0): hidden element count + iframe count.
    Tier 2 (only with include_page_info): shadow DOM, images, links, compaction, depth.
    """
    lines: list[str] = []

    hidden = page_stats.get('hidden', {})
    hidden_total = hidden.get('total', 0)
    if hidden_total > 0:
        reasons = []
        for key, label in hidden_reason_keys:
            count = hidden.get(key, 0)
            if count > 0:
                reasons.append(f'{label}: {count}')
        lines.append(f'# hidden: {hidden_total} ({", ".join(reasons)})')

    iframe_count = page_stats.get('iframes', 0)
    if iframe_count > 0:
        lines.append(f'# iframes: {iframe_count} (content not traversed)')

    if include_page_info:
        shadow_count = page_stats.get('shadowRoots', 0)
        if shadow_count > 0:
            lines.append(f'# shadow_roots: {shadow_count} (content not traversed)')

        images = page_stats.get('images', {})
        img_total = images.get('total', 0)
        if img_total > 0:
            lines.append(
                f'# images: {img_total} '
                f'(with_alt: {images.get("withAlt", 0)}, without_alt: {images.get("withoutAlt", 0)})',
            )

        link_count = page_stats.get('links', 0)
        if link_count > 0 and not include_urls:
            lines.append(f'# links: {link_count} (urls not included \u2014 use include_urls=True)')

        if raw_node_count > 0 and compact_tree and raw_node_count != compacted_node_count:
            lines.append(f'# tree: {raw_node_count} nodes \u2192 {compacted_node_count} after compaction')

        depth_trunc = page_stats.get('depthTruncated', 0)
        if depth_trunc > 0:
            lines.append(f'# depth_truncated: {depth_trunc} subtrees cut at depth 50')

    if not lines:
        return ''
    return '\n# --- page metadata ---\n' + '\n'.join(lines)


def save_large_output_to_file(content: str, output_dir: Path, prefix: str, extension: str) -> str:
    """Save large output to file, return formatted response with path and preview.

    Preserves natural line structure by writing directly to disk,
    bypassing MCP JSON serialization that would escape newlines.
    """
    timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{prefix}_{timestamp}.{extension}'
    file_path = output_dir / filename

    file_path.write_text(content, encoding='utf-8')

    line_count = content.count('\n') + 1
    char_count = len(content)
    preview = content[:2000]

    return (
        f'{prefix.replace("_", " ").title()} '
        f'({char_count:,} chars, {line_count:,} lines) saved to:\n'
        f'{file_path}\n\n'
        f'Preview (first 2000 chars):\n{preview}'
    )


def build_storage_init_script(profile_state: ProfileState) -> str | None:
    """Build JavaScript init script to restore localStorage/sessionStorage.

    Creates a script that runs via Page.addScriptToEvaluateOnNewDocument BEFORE
    any page JavaScript executes. This ensures storage is populated before apps
    check for auth tokens, user preferences, or other initialization data.
    """
    storage_data: dict[str, dict[str, dict[str, str]]] = {}

    for origin_url, origin_data in profile_state.origins.items():
        has_local = origin_data.local_storage and len(origin_data.local_storage) > 0
        has_session = origin_data.session_storage and len(origin_data.session_storage) > 0

        if has_local or has_session:
            storage_data[origin_url] = {
                'localStorage': dict(origin_data.local_storage) if origin_data.local_storage else {},
                'sessionStorage': dict(origin_data.session_storage) if origin_data.session_storage else {},
            }

    if not storage_data:
        return None

    storage_json = json.dumps(storage_data, ensure_ascii=False)

    # fmt: off
    return f"""\
(function() {{
    'use strict';

    window.__storageRestoreState = {{
        restored: {{ localStorage: [], sessionStorage: [] }},
        errors: [],
        origin: null,
        timestamp: Date.now()
    }};

    var storageData = {storage_json};
    var currentOrigin = window.location.origin;
    window.__storageRestoreState.origin = currentOrigin;

    var data = storageData[currentOrigin];
    if (!data) {{
        return;
    }}

    if (data.localStorage) {{
        Object.keys(data.localStorage).forEach(function(key) {{
            try {{
                window.localStorage.setItem(key, data.localStorage[key]);
                window.__storageRestoreState.restored.localStorage.push(key);
            }} catch (e) {{
                window.__storageRestoreState.errors.push({{
                    type: 'localStorage',
                    key: key,
                    error: e.name,
                    message: e.message
                }});
            }}
        }});
    }}

    if (data.sessionStorage) {{
        Object.keys(data.sessionStorage).forEach(function(key) {{
            try {{
                window.sessionStorage.setItem(key, data.sessionStorage[key]);
                window.__storageRestoreState.restored.sessionStorage.push(key);
            }} catch (e) {{
                window.__storageRestoreState.errors.push({{
                    type: 'sessionStorage',
                    key: key,
                    error: e.name,
                    message: e.message
                }});
            }}
        }});
    }}
}})();
"""
    # fmt: on
