"""Pure tree transformation functions for ARIA and visual tree processing.

These functions handle tree compaction and serialization independent of browser interaction.
Extracted from server.py to enable unit testing and reuse.
"""

from __future__ import annotations

import unicodedata
from typing import Any

__all__ = [
    'compact_aria_tree',
    'compact_visual_tree',
    'serialize_aria_snapshot',
    'serialize_visual_tree',
]


def compact_aria_tree(node: dict[str, Any] | None) -> dict[str, Any] | None:
    """Recursively compact tree by removing structural noise.

    Bottom-up recursion ensures children are compacted before parent decisions.
    Returns None for nodes that should be removed, otherwise a new (not mutated) node.

    Compaction rules:
    1. Remove empty generics (no name, description, or children)
    2. Collapse single-child generic chains (promote the child)
    3. Remove redundant text children when:
       - ALL children are text nodes, AND
       - Their space-joined concatenation equals the element's name
       - Comparison uses NFKC normalization for Unicode equivalence
    """
    if node is None:
        return None

    # Text nodes pass through unchanged
    if node.get('type') == 'text':
        return node

    # Process children first (bottom-up recursion)
    children = node.get('children', [])
    compacted_children = []
    for child in children:
        compacted = compact_aria_tree(child)
        if compacted is not None:
            compacted_children.append(compacted)

    role = node.get('role', 'generic')
    name = node.get('name', '')
    has_description = bool(node.get('description'))

    # Preserve containers with visibility markers - don't collapse them
    has_visibility_marker = bool(node.get('hidden') or node.get('visuallyHidden'))

    # Rule 1: Remove empty generics (no semantic content, no children)
    # Exception: Keep nodes with visibility markers even if empty (debugging value)
    if role == 'generic' and not name and not has_description and not compacted_children and not has_visibility_marker:
        return None

    # Rule 2: Collapse single-child generic chains (unwrap wrapper divs)
    # Exception: Don't collapse containers with visibility markers (preserve structure)
    if (
        role == 'generic'
        and not name
        and not has_description
        and len(compacted_children) == 1
        and not has_visibility_marker
    ):
        return compacted_children[0]

    # Rule 3: Remove redundant text children (name already captures the text)
    # Enhanced: handles multiple text children via concatenation
    if name and compacted_children:
        all_text = all(c.get('type') == 'text' for c in compacted_children)
        if all_text:
            texts = [c.get('content', '') for c in compacted_children]
            concatenated = ' '.join(texts)
            # Use NFKC normalization for comparison (handles ellipsis, etc.)
            if _normalize_for_comparison(concatenated) == _normalize_for_comparison(name):
                compacted_children = []

    # Return node with updated children (don't mutate original)
    result = {k: v for k, v in node.items() if k != 'children'}
    if compacted_children:
        result['children'] = compacted_children
    return result


def compact_visual_tree(node: dict[str, Any]) -> dict[str, Any] | None:
    """Recursively compact visual tree by removing structural noise.

    Same compaction rules as compact_aria_tree but for visual tree format.
    Only difference: checks 'hidden' instead of both 'hidden' and 'visuallyHidden'.
    """
    if node.get('type') == 'text':
        return node

    children = node.get('children', [])
    compacted_children = []
    for child in children:
        compacted = compact_visual_tree(child)
        if compacted is not None:
            compacted_children.append(compacted)

    role = node.get('role', 'generic')
    name = node.get('name', '')
    has_description = bool(node.get('description'))
    has_visibility_marker = bool(node.get('hidden'))

    # Rule 1: Remove empty generics
    if role == 'generic' and not name and not has_description and not compacted_children and not has_visibility_marker:
        return None

    # Rule 2: Collapse single-child generic chains
    if (
        role == 'generic'
        and not name
        and not has_description
        and len(compacted_children) == 1
        and not has_visibility_marker
    ):
        return compacted_children[0]

    # Rule 3: Remove redundant text children
    if name and compacted_children:
        all_text = all(c.get('type') == 'text' for c in compacted_children)
        if all_text:
            texts = [c.get('content', '') for c in compacted_children]
            concatenated = ' '.join(texts)
            if _normalize_for_comparison(concatenated) == _normalize_for_comparison(name):
                compacted_children = []

    # Return node with updated children (don't mutate original)
    result = {k: v for k, v in node.items() if k != 'children'}
    if compacted_children:
        result['children'] = compacted_children
    return result


def serialize_aria_snapshot(node: dict[str, Any] | None, indent: int = 0) -> str:
    """Custom serializer matching Playwright's ARIA snapshot YAML format."""
    if node is None:
        return ''

    lines = []
    prefix = ' ' * indent + '- '

    # Handle text nodes
    if node.get('type') == 'text':
        # Full whitespace normalization: collapse all \s+ to single space
        content = ' '.join(node.get('content', '').split())
        lines.append(f'{prefix}text: {content}')
        return '\n'.join(lines)

    # Handle element nodes
    role = node.get('role', 'generic')
    name = node.get('name', '')
    children = node.get('children', [])

    # Build node header in Playwright format: role "name" [attrs]:
    header = f'{prefix}{role}'

    if name:
        # Escape quotes in name
        escaped_name = name.replace('"', '\\"')
        header += f' "{escaped_name}"'

    # Add attributes in brackets
    attrs = []
    if 'level' in node:
        attrs.append(f'level={node["level"]}')

    # Checkbox/radio/switch checked state (always show for these roles)
    if 'checked' in node:
        val = node['checked']
        if val == 'mixed':
            attrs.append('checked=mixed')
        elif val:
            attrs.append('checked')
        else:
            attrs.append('unchecked')

    # Selected state for tabs, options, treeitems, etc.
    if 'selected' in node:
        if node['selected']:
            attrs.append('selected')
        else:
            attrs.append('selected=false')

    # Pressed state for toggle buttons
    if 'pressed' in node:
        val = node['pressed']
        if val == 'mixed':
            attrs.append('pressed=mixed')
        elif val:
            attrs.append('pressed')
        else:
            attrs.append('pressed=false')

    # Expanded state for disclosure widgets, accordions, etc.
    if 'expanded' in node:
        if node['expanded']:
            attrs.append('expanded')
        else:
            attrs.append('expanded=false')

    if node.get('disabled'):
        attrs.append('disabled')
    if node.get('url'):
        attrs.append(f'url={node["url"]}')
    # Add hidden marker (element NOT in accessibility tree)
    if node.get('hidden'):
        attrs.append(f'hidden:{node["hidden"]}')
    # Add visually-hidden marker (element IN accessibility tree but not visible)
    if node.get('visuallyHidden'):
        attrs.append(f'visually-hidden:{node["visuallyHidden"]}')

    if attrs:
        header += f' [{", ".join(attrs)}]'

    # Add colon if has children
    if children:
        header += ':'

    lines.append(header)

    # Process children with increased indentation
    if children:
        for child in children:
            child_output = serialize_aria_snapshot(child, indent + 2)
            if child_output:
                lines.append(child_output)

    return '\n'.join(lines)


def serialize_visual_tree(node: dict[str, Any] | None, indent: int = 0) -> str:
    """Serialize visual tree to YAML format (same structure as ARIA snapshot)."""
    if node is None:
        return ''

    prefix = ' ' * indent

    if node.get('type') == 'text':
        return f'{prefix}- text: {node.get("content", "")}'

    role = node.get('role', 'generic')
    name = node.get('name', '')

    attrs: list[str] = []
    if node.get('hidden'):
        attrs.append(f'hidden:{node["hidden"]}')
    if node.get('level'):
        attrs.append(f'level={node["level"]}')

    # Checkbox/radio/switch checked state (always show for these roles)
    if 'checked' in node:
        val = node['checked']
        if val == 'mixed':
            attrs.append('checked=mixed')
        elif val:
            attrs.append('checked')
        else:
            attrs.append('unchecked')

    # Selected state for tabs, options, treeitems, etc.
    if 'selected' in node:
        if node['selected']:
            attrs.append('selected')
        else:
            attrs.append('selected=false')

    # Pressed state for toggle buttons
    if 'pressed' in node:
        val = node['pressed']
        if val == 'mixed':
            attrs.append('pressed=mixed')
        elif val:
            attrs.append('pressed')
        else:
            attrs.append('pressed=false')

    # Expanded state for disclosure widgets, accordions, etc.
    if 'expanded' in node:
        if node['expanded']:
            attrs.append('expanded')
        else:
            attrs.append('expanded=false')

    if node.get('disabled'):
        attrs.append('disabled')
    if node.get('url'):
        attrs.append(f'url={node["url"]}')

    attr_str = f' [{", ".join(attrs)}]' if attrs else ''
    name_str = f' "{name}"' if name else ''

    children = node.get('children', [])
    if children:
        lines = [f'{prefix}- {role}{name_str}{attr_str}:']
        for child in children:
            child_output = serialize_visual_tree(child, indent + 2)
            if child_output:
                lines.append(child_output)
        return '\n'.join(lines)
    else:
        return f'{prefix}- {role}{name_str}{attr_str}'


def _normalize_for_comparison(s: str) -> str:
    """Normalize using NFKC for visually-equivalent character comparison.

    NFKC (Compatibility Composition) normalizes:
    - Ellipsis (…) → three periods (...)
    - Ligatures (ﬁ) → component characters (fi)
    - Some other compatibility equivalents

    Note: NFKC does NOT normalize curly quotes or dashes to ASCII equivalents.
    Those remain distinct. This is intentional - only true compatibility
    equivalents are normalized.
    """
    return unicodedata.normalize('NFKC', s)
