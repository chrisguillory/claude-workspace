from __future__ import annotations

__all__ = [
    'register_extraction_tools',
]

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    PageTextResult,
)
from ..service import BrowserService


def register_extraction_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register extraction tools."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Get Page Text',
            readOnlyHint=True,
            openWorldHint=True,
        ),
    )
    async def get_page_text(
        ctx: Context[Any, Any, Any],
        selector: str = 'auto',
        include_images: bool = False,
    ) -> PageTextResult:
        """Extract all text content from page, including hidden content.

        **Default behavior (selector='auto'):**
        Smart extraction tries semantic elements in priority order:
        1. `<main>` element (if >500 chars)
        2. `<article>` element (if >500 chars)
        3. Falls back to `<body>`

        **Explicit extraction:**
        Use `selector='body'` for full page content, or any CSS selector
        for specific elements.

        **Image alt text (include_images):**
        When True, includes image descriptions inline as `[Image: alt text]`.
        Images without alt text appear as `[Image: (no alt)]`.

        **Transparency metadata (auto mode only):**
        For selector='auto', response includes `smart_info` with:
        - fallback_used: True if no suitable main/article found, fell back to body
        - body_character_count: Total body chars for coverage calculation

        Coverage ratio: `character_count / smart_info.body_character_count`

        For explicit selectors, `smart_info` is None (not applicable).

        The extraction automatically:
        - Filters out script, style, template, and other non-content elements
        - Normalizes whitespace while preserving paragraph structure
        - Traverses into Shadow DOM components
        - Preserves whitespace in PRE/CODE/TEXTAREA elements

        Args:
            selector: 'auto' (default, smart extraction) or CSS selector
                      Common values:
                      - 'auto' - smart extraction (default)
                      - 'body' - full page
                      - 'main' - main content area
                      - 'article' - article content
            include_images: Include image alt text as [Image: description]

        Returns:
            PageTextResult with text, metadata, and transparency fields

        Examples:
            # Smart extraction (default)
            result = get_page_text()

            # Full page extraction (explicit)
            result = get_page_text(selector="body")

            # Include image descriptions
            result = get_page_text(include_images=True)

        Notes:
            - For HTML source, use get_page_html() instead
            - For page structure, use get_aria_snapshot() first
            - Iframe content is not extracted (matches Chrome behavior)
        """
        return await service.get_page_text(selector=selector, include_images=include_images)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Get Page HTML',
            readOnlyHint=True,
            openWorldHint=True,
        ),
    )
    async def get_page_html(ctx: Context[Any, Any, Any], selector: str | None = None, limit: int | None = None) -> str:
        """Get raw HTML source or specific elements.

        Use this when you need actual HTML markup for inspection or parsing.
        For readable text content, use get_page_text() instead.

        Args:
            selector: CSS selector to match elements (optional)
                      If not provided, returns full page source
            limit: Maximum number of elements when using selector

        Returns:
            - If no selector: Full page HTML source
            - If selector: outerHTML of matched elements, separated by newlines

        Examples:
            # Full page HTML source
            html = get_page_html()

            # All article elements
            html = get_page_html(selector="article")

            # First 3 list items
            html = get_page_html(selector="li", limit=3)

            # Specific form by ID
            html = get_page_html(selector="form#login")
        """
        return await service.get_page_html(selector=selector, limit=limit)

    @mcp.tool(annotations=ToolAnnotations(title='Get ARIA Snapshot', readOnlyHint=True))
    async def get_aria_snapshot(
        selector: str,
        include_urls: bool = False,
        compact_tree: bool = True,
        include_hidden: bool = False,
        include_page_info: bool = False,
    ) -> str:
        """Understand page structure and find elements. PRIMARY tool for page comprehension.

        Returns semantic accessibility tree with roles, names, and hierarchy. Use immediately
        after navigate() or any action that changes page content. Provides complete structural
        understanding with 91% token compression vs HTML.

        Args:
            selector: CSS selector scope ('body' for full page, 'form' for specific sections)
            include_urls: Include href values (default False saves tokens)
            compact_tree: Remove structural noise from tree (default True). Applies:
                1. Remove empty generics (divs with no content)
                2. Collapse single-child generic chains (unwrap wrapper divs)
                3. Remove redundant text children (when name equals text content)
                Set to False for complete unfiltered tree.
            include_hidden: Include hidden content with markers (default False).
                When False: Excludes aria-hidden, inert, and CSS-hidden elements.
                    Matches what assistive technology perceives.
                When True: Includes ALL content with [hidden:X] markers showing
                    hiding mechanism (aria, inert, css, or combinations).
                    Use for debugging "why can't I find this element?" or
                    seeing collapsed menus, hidden modals, full page structure.
            include_page_info: Show extended page statistics (shadow roots, images, links,
                compaction ratio). Hidden element and iframe counts always appear when > 0.

        Returns:
            YAML with ARIA roles, accessible names, element hierarchy, and states.
            When include_hidden=True, hidden elements show [hidden:X] markers.
            Always appends hidden element and iframe counts (when > 0) as YAML comments.

        Workflow:
            1. Call this after navigate() to understand available elements
            2. Use returned structure to identify elements of interest
            3. Call get_interactive_elements() if you need CSS selectors for clicking
            4. Call get_page_text() if you need actual text content
        """
        return await service.get_aria_snapshot(
            selector=selector,
            include_urls=include_urls,
            compact_tree=compact_tree,
            include_hidden=include_hidden,
            include_page_info=include_page_info,
        )

    @mcp.tool(annotations=ToolAnnotations(title='Get Visual Tree', readOnlyHint=True))
    async def get_visual_tree(
        selector: str,
        include_urls: bool = False,
        compact_tree: bool = True,
        include_hidden: bool = False,
        include_page_info: bool = False,
    ) -> str:
        """Show what sighted users see. Complements get_aria_snapshot() (what AT sees).

        Key difference from ARIA snapshot:
        - aria-hidden elements ARE included (they're visible to sighted users!)
        - opacity:0 elements are EXCLUDED (invisible to sighted users)
        - sr-only/clipped elements are EXCLUDED (invisible to sighted users)

        Use cases:
        - Visual verification: understanding what users actually see
        - Screenshot correlation: matching DOM to visual regions
        - UI testing: verifying visual state
        - Debugging: "why can user see this but screen reader can't?"

        Args:
            selector: CSS selector scope ('body' for full page)
            include_urls: Include href values (default False)
            compact_tree: Remove structural noise (default True)
            include_hidden: Include visually hidden content with markers (default False)
            include_page_info: Show extended page statistics (shadow roots, images, links,
                compaction ratio). Hidden element and iframe counts always appear when > 0.

        Returns:
            YAML tree of visually rendered elements.
            When include_hidden=True, hidden elements show [hidden:X] markers.
            Always appends hidden element and iframe counts (when > 0) as YAML comments.

        Comparison:
            | Mechanism       | ARIA Tree  | Visual Tree |
            |-----------------|------------|-------------|
            | aria-hidden     | EXCLUDED   | INCLUDED    |
            | opacity:0       | INCLUDED   | EXCLUDED    |
            | sr-only/clip    | INCLUDED   | EXCLUDED    |
            | display:none    | EXCLUDED   | EXCLUDED    |
        """
        return await service.get_visual_tree(
            selector=selector,
            include_urls=include_urls,
            compact_tree=compact_tree,
            include_hidden=include_hidden,
            include_page_info=include_page_info,
        )
