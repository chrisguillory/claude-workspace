"""Unit tests for _count_tree_nodes and _build_page_metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from selenium_browser_automation.server import (
    _ARIA_HIDDEN_REASON_KEYS,
    _VISUAL_HIDDEN_REASON_KEYS,
    _build_page_metadata,
    _count_tree_nodes,
)

# --- _count_tree_nodes ---


class TestCountTreeNodes:
    def test_none_returns_zero(self) -> None:
        assert _count_tree_nodes(None) == 0

    def test_single_node(self) -> None:
        assert _count_tree_nodes({'role': 'button', 'name': 'Click'}) == 1

    def test_text_node(self) -> None:
        assert _count_tree_nodes({'type': 'text', 'content': 'hello'}) == 1

    def test_parent_with_children(self) -> None:
        tree: dict[str, Any] = {
            'role': 'list',
            'children': [
                {'role': 'listitem', 'name': 'A'},
                {'role': 'listitem', 'name': 'B'},
                {'role': 'listitem', 'name': 'C'},
            ],
        }
        assert _count_tree_nodes(tree) == 4  # parent + 3 children

    def test_nested_tree(self) -> None:
        tree: dict[str, Any] = {
            'role': 'navigation',
            'children': [
                {
                    'role': 'list',
                    'children': [
                        {'role': 'listitem', 'children': [{'type': 'text', 'content': 'Home'}]},
                        {'role': 'listitem', 'children': [{'type': 'text', 'content': 'About'}]},
                    ],
                }
            ],
        }
        # nav(1) > list(1) > [listitem(1) > text(1), listitem(1) > text(1)] = 6
        assert _count_tree_nodes(tree) == 6

    def test_no_children_key(self) -> None:
        assert _count_tree_nodes({'role': 'img', 'name': 'Logo'}) == 1

    def test_empty_children(self) -> None:
        assert _count_tree_nodes({'role': 'div', 'children': []}) == 1


# --- _build_page_metadata ---


def _meta(
    page_stats: dict[str, Any] | None = None,
    include_page_info: bool = False,
    include_urls: bool = False,
    compact_tree: bool = False,
    raw_node_count: int = 0,
    compacted_node_count: int = 0,
    hidden_reason_keys: Sequence[tuple[str, str]] = _ARIA_HIDDEN_REASON_KEYS,
) -> str:
    """Helper to call _build_page_metadata with sensible defaults."""
    return _build_page_metadata(
        page_stats=page_stats or {},
        include_page_info=include_page_info,
        include_urls=include_urls,
        compact_tree=compact_tree,
        raw_node_count=raw_node_count,
        compacted_node_count=compacted_node_count,
        hidden_reason_keys=hidden_reason_keys,
    )


class TestBuildPageMetadataTier1:
    """Tier 1: Always-on metadata (hidden elements, iframes)."""

    def test_empty_stats(self) -> None:
        assert _meta() == ''

    def test_hidden_elements(self) -> None:
        stats = {'hidden': {'total': 5, 'ariaHidden': 3, 'inert': 2}}
        result = _meta(page_stats=stats)
        assert '# hidden: 5 (aria-hidden: 3, inert: 2)' in result

    def test_hidden_skips_zero_reasons(self) -> None:
        stats = {'hidden': {'total': 3, 'ariaHidden': 3, 'inert': 0}}
        result = _meta(page_stats=stats)
        assert 'inert' not in result
        assert 'aria-hidden: 3' in result

    def test_iframes(self) -> None:
        result = _meta(page_stats={'iframes': 2})
        assert '# iframes: 2' in result

    def test_zero_hidden_suppressed(self) -> None:
        result = _meta(page_stats={'hidden': {'total': 0}})
        assert result == ''

    def test_zero_iframes_suppressed(self) -> None:
        result = _meta(page_stats={'iframes': 0})
        assert result == ''

    def test_visual_hidden_reasons(self) -> None:
        stats = {'hidden': {'total': 4, 'opacity': 2, 'clipped': 1, 'offscreen': 1}}
        result = _meta(page_stats=stats, hidden_reason_keys=_VISUAL_HIDDEN_REASON_KEYS)
        assert 'opacity: 2' in result
        assert 'clipped: 1' in result
        assert 'offscreen: 1' in result


class TestBuildPageMetadataTier2:
    """Tier 2: Only with include_page_info=True."""

    def test_shadow_roots(self) -> None:
        result = _meta(
            page_stats={'shadowRoots': 3},
            include_page_info=True,
        )
        assert '# shadow_roots: 3' in result

    def test_shadow_roots_hidden_without_flag(self) -> None:
        result = _meta(page_stats={'shadowRoots': 3}, include_page_info=False)
        assert result == ''

    def test_images(self) -> None:
        result = _meta(
            page_stats={'images': {'total': 10, 'withAlt': 7, 'withoutAlt': 3}},
            include_page_info=True,
        )
        assert '# images: 10' in result
        assert 'with_alt: 7' in result
        assert 'without_alt: 3' in result

    def test_links_shown_when_urls_excluded(self) -> None:
        result = _meta(
            page_stats={'links': 15},
            include_page_info=True,
            include_urls=False,
        )
        assert '# links: 15' in result
        assert 'include_urls=True' in result

    def test_links_hidden_when_urls_included(self) -> None:
        result = _meta(
            page_stats={'links': 15},
            include_page_info=True,
            include_urls=True,
        )
        assert 'links' not in result

    def test_compaction_stats(self) -> None:
        result = _meta(
            include_page_info=True,
            compact_tree=True,
            raw_node_count=30,
            compacted_node_count=18,
        )
        assert '# tree: 30 nodes' in result
        assert '18 after compaction' in result

    def test_compaction_suppressed_when_no_change(self) -> None:
        result = _meta(
            include_page_info=True,
            compact_tree=True,
            raw_node_count=10,
            compacted_node_count=10,
        )
        assert 'tree' not in result

    def test_compaction_suppressed_when_compact_false(self) -> None:
        result = _meta(
            include_page_info=True,
            compact_tree=False,
            raw_node_count=30,
            compacted_node_count=18,
        )
        assert 'tree' not in result

    def test_depth_truncation(self) -> None:
        result = _meta(
            page_stats={'depthTruncated': 2},
            include_page_info=True,
        )
        assert '# depth_truncated: 2' in result

    def test_metadata_header(self) -> None:
        result = _meta(page_stats={'iframes': 1})
        assert result.startswith('\n# --- page metadata ---\n')


class TestBuildPageMetadataCombined:
    """Tier 1 + Tier 2 together."""

    def test_both_tiers(self) -> None:
        stats: dict[str, Any] = {
            'hidden': {'total': 3, 'ariaHidden': 3},
            'iframes': 1,
            'shadowRoots': 2,
            'images': {'total': 5, 'withAlt': 5, 'withoutAlt': 0},
            'links': 10,
        }
        result = _meta(page_stats=stats, include_page_info=True)
        assert '# hidden: 3' in result
        assert '# iframes: 1' in result
        assert '# shadow_roots: 2' in result
        assert '# images: 5' in result
        assert '# links: 10' in result
