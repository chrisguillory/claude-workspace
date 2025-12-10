#!/usr/bin/env python3
"""
Browser Automation MCP Server

Playwright-based browser control with stealth mode for Claude Code.

Install:
    uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/browser-automation server

Architecture: Runs locally (not Docker) for visible browser monitoring.
"""

from __future__ import annotations

# Standard Library
import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

# Third-Party Libraries
import fastmcp.exceptions
import yaml
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from local_lib.utils import DualLogger
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel

"""
Architecture: All tools operate on singleton _page (shared browser tab state).
Workflow: navigate(url) → get_page_content() → click() → screenshot()

Resource Capture: Must happen DURING navigation (route handlers before page.goto).
Therefore integrated into navigate(capture_resources=True) not separate tool.
- navigate(capture_resources=True): Captures all resources during page load
- download_resource(url, filename): Downloads specific resource using page context

Browser Context: All requests share session/cookies - bypasses bot detection.
Temp Files: Auto-cleanup on shutdown via _temp_dir.
"""


@asynccontextmanager
async def lifespan(server):
    """Manage browser lifecycle - cleanup on shutdown."""
    # Browser is lazy-initialized on first use
    try:
        yield {}
    finally:
        # Cleanup: browser.close() closes all contexts/pages automatically
        global _playwright, _browser
        if _browser:
            await _browser.close()
        if _playwright:
            await _playwright.stop()


mcp = FastMCP("browser-automation", lifespan=lifespan)


# Pydantic models for structured output
class CapturedResource(BaseModel):
    url: str
    path: str
    absolute_path: str
    type: str
    size_bytes: int
    content_type: str
    status: int


class ResourceCapture(BaseModel):
    output_dir: str
    html_path: str
    captured: list[CapturedResource]
    total_size_mb: float
    resource_count: int
    errors: list[dict]


class NavigationResult(BaseModel):
    current_url: str
    title: str
    resources: ResourceCapture | None = None


class InteractiveElement(BaseModel):
    tag: str
    text: str
    selector: str
    cursor: str
    href: str | None
    classes: str


class FocusableElement(BaseModel):
    tag: str
    text: str
    selector: str
    tab_index: int
    is_tabbable: bool
    classes: str


# Browser session state (lazy initialized)
_playwright = None
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None

# Temporary directory for screenshots (auto-cleanup on process exit)
_temp_dir = tempfile.TemporaryDirectory()
_screenshot_dir = Path(_temp_dir.name)

# Temporary directory for resource captures (auto-cleanup on process exit)
_capture_temp_dir = tempfile.TemporaryDirectory()
_capture_dir = Path(_capture_temp_dir.name)
_capture_counter = 0  # Zero-padded 3-digit counter for capture directories


def _save_captured_resource(
    capture_dir: Path, url: str, data: dict
) -> CapturedResource | None:
    """Save single captured resource to hierarchical path. Returns metadata or None on error."""
    # Create hierarchical path from URL
    parsed = urlparse(url)
    rel_path = Path(parsed.netloc) / parsed.path.lstrip("/")

    # Sanitize filename (last component only)
    parts = list(rel_path.parts)
    parts[-1] = "".join(c if c.isalnum() or c in ".-_" else "_" for c in parts[-1])
    if not parts[-1]:
        parts[-1] = "resource"
    rel_path = Path(*parts) if len(parts) > 1 else Path(parts[0])

    abs_path = capture_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    # Save resource
    abs_path.write_bytes(data["body"])

    return CapturedResource(
        url=url,
        path=str(rel_path),
        absolute_path=str(abs_path),
        type=data["type"],
        size_bytes=data["size"],
        content_type=data["content_type"],
        status=data["status"],
    )


def _remove_url_fields(node):
    """Recursively remove /url fields and empty containers from YAML data structure."""
    if isinstance(node, dict):
        filtered = {k: _remove_url_fields(v) for k, v in node.items() if k != "/url"}
        # Remove keys with empty/None values
        filtered = {k: v for k, v in filtered.items() if v not in (None, [], {})}
        return filtered if filtered else None
    elif isinstance(node, list):
        filtered = [_remove_url_fields(item) for item in node]
        # Filter out None and empty containers
        filtered = [item for item in filtered if item not in (None, [], {})]
        return filtered if filtered else None
    else:
        return node


async def _close_browser():
    """Tear down browser, context, and page. Reset all global state."""
    global _playwright, _browser, _context, _page

    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()

    _page = None
    _context = None
    _browser = None
    _playwright = None


async def get_browser() -> tuple[Browser, BrowserContext, Page]:
    """Initialize and return browser session (lazy singleton pattern)."""
    global _playwright, _browser, _context, _page

    if _page is not None:
        return _browser, _context, _page

    # Third-Party Libraries
    from playwright_stealth.stealth import Stealth

    _playwright = await async_playwright().start()

    _browser = await _playwright.chromium.launch(
        headless=False,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--window-size=1920,1080",
        ],
    )

    _context = await _browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        locale="en-US",
        timezone_id="America/Los_Angeles",
    )

    _page = await _context.new_page()
    stealth_config = Stealth()
    await stealth_config.apply_stealth_async(_page)

    return _browser, _context, _page


@mcp.tool(
    annotations=ToolAnnotations(
        title="Navigate to URL",
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def navigate(
    url: str,
    capture_resources: bool = False,
    resource_types: list[str] | None = None,
    max_size_mb: float = 50.0,
    fresh_browser: bool = False,
) -> NavigationResult:
    """Navigate to URL. Optionally capture all resources (JS, CSS, images) during load.

    Route handlers must be set BEFORE navigation, so capture integrated here not separate tool.
    All requests share browser session/cookies - bypasses bot detection (e.g., SEC.gov).

    Args:
        url: Full URL (http:// or https://)
        capture_resources: If True, capture all page resources during navigation
        resource_types: Filter types - ['script', 'stylesheet', 'image', 'font', 'document', 'xhr', 'fetch']
            None = all types. Only used if capture_resources=True.
        max_size_mb: Skip resources larger than this (per-resource). Default: 50MB.
        fresh_browser: If True, tear down and recreate browser for clean session (no cache/cookies)

    Returns:
        NavigationResult with current_url, title, and optional resources field.
        When capture_resources=True, resources field contains output_dir, captured list, etc.

    Organization: Hierarchical by domain/path. HTML as index.html in output_dir.
    Errors: Navigation errors raise ToolError. Resource capture errors non-fatal (partial success).
    """
    if not url.startswith(("http://", "https://")):
        raise fastmcp.exceptions.ValidationError(
            "URL must start with http:// or https://"
        )

    print(
        f"[navigate] Navigating to {url}"
        + (" with resource capture" if capture_resources else "")
        + (" (fresh browser)" if fresh_browser else "")
    )

    if fresh_browser:
        await _close_browser()

    _, _, page = await get_browser()

    captured_resources = {}
    capture_dir = None

    if capture_resources:
        global _capture_counter
        _capture_counter += 1
        capture_dir = _capture_dir / f"capture_{_capture_counter:03d}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        # Response event monitoring - doesn't block network stack
        async def capture_response(response):
            try:
                req_type = response.request.resource_type

                # Apply type filter
                if resource_types and req_type not in resource_types:
                    return

                # Check status
                if response.status >= 400:
                    return

                # Check size from headers first
                content_length = response.headers.get("content-length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        return

                # Fetch body
                body = await response.body()

                captured_resources[response.url] = {
                    "body": body,
                    "type": req_type,
                    "content_type": response.headers.get("content-type", "unknown"),
                    "status": response.status,
                    "size": len(body),
                }
            except Exception as e:
                captured_resources[response.url] = {"error": str(e)}

        page.on("response", lambda r: asyncio.create_task(capture_response(r)))

    # Navigate with 'load' instead of 'networkidle' (more reliable)
    response = await page.goto(url, wait_until="load", timeout=30000)

    if response is None:
        raise fastmcp.exceptions.ToolError("Navigation failed: no response")

    # Wait for async resources to complete
    if capture_resources:
        await asyncio.sleep(2)

    print(f"[navigate] Successfully navigated to {page.url}")

    result = NavigationResult(current_url=page.url, title=await page.title())

    if capture_resources:
        # Save main HTML
        html_path = capture_dir / "index.html"
        html_path.write_text(await page.content(), encoding="utf-8")

        # Process and save resources
        captured_list = []
        errors = []
        total_size = 0

        for res_url, res_data in captured_resources.items():
            if "error" in res_data:
                errors.append({"url": res_url, "error": res_data["error"]})
                continue

            saved = _save_captured_resource(capture_dir, res_url, res_data)
            if saved:
                captured_list.append(saved)
                total_size += saved.size_bytes
            else:
                errors.append({"url": res_url, "error": "failed to save"})

        result.resources = ResourceCapture(
            output_dir=str(capture_dir),
            html_path=str(html_path),
            captured=captured_list,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            resource_count=len(captured_list),
            errors=errors,
        )

        print(
            f"[navigate] Captured {len(captured_list)} resources ({result.resources.total_size_mb}MB)"
        )

    return result


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Page Content", readOnlyHint=True, openWorldHint=True
    )
)
async def get_page_content(
    format: Literal["text", "html", "markdown"],
    ctx: Context,
    selector: str | None = None,
    limit: int | None = None,
) -> str:
    """Extract page content. Prefer "text" over "html" for efficiency.

    Args:
        format: Output format
            - "text": Plain text (recommended)
            - "html": Raw HTML (large, use sparingly)
            - "markdown": Not yet implemented
        ctx: MCP context
        selector: CSS selector to limit extraction (e.g., 'img', 'main', '.content')
            When provided, returns matching elements instead of full page
        limit: Max number of elements to return when using selector (None = all matches)

    Returns:
        Page content as string. When selector is used, returns outer HTML of matched elements.
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(
        f"Extracting as {format}" + (f' with selector "{selector}"' if selector else "")
    )

    if selector:
        # Extract specific elements
        locator = page.locator(selector)
        count = await locator.count()

        actual_limit = min(count, limit) if limit is not None else count

        html_parts = []
        for i in range(actual_limit):
            html = await locator.nth(i).evaluate("el => el.outerHTML")
            html_parts.append(html)

        await logger.info(f"Matched {len(html_parts)} elements")
        return "\n".join(html_parts)

    # Full page extraction
    if format == "text":
        content = await page.inner_text("body")
        return content
    elif format == "html":
        content = await page.content()
        return content
    elif format == "markdown":
        # TODO: implement markdown conversion (could use html2text library)
        raise fastmcp.exceptions.ValidationError("Markdown format not yet implemented")
    else:
        raise fastmcp.exceptions.ValidationError(f"Unknown format: {format}")


@mcp.tool(annotations=ToolAnnotations(title="Take Screenshot", readOnlyHint=True))
async def screenshot(filename: str, ctx: Context) -> str:
    """Take full-page screenshot. Saves to temp directory with auto-cleanup on exit.

    Args:
        filename: Name for screenshot file (e.g., "homepage.png")
        ctx: MCP context

    Returns:
        Absolute path to temp file (use Read tool to view image)
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(f"Taking screenshot: {filename}")

    screenshot_path = _screenshot_dir / filename
    await page.screenshot(path=str(screenshot_path), full_page=True)

    await logger.info(f"Screenshot saved to {screenshot_path}")
    return str(screenshot_path)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Download Specific Resource", readOnlyHint=False, idempotentHint=False
    )
)
async def download_resource(url: str, output_filename: str) -> dict:
    """Download specific resource using current browser context and session.

    Uses page.request.get() to maintain cookies/session from prior navigation.
    Critical for sites with bot detection - site sees this as same browser session.

    PREREQUISITE: Call navigate() first to establish browser session.
    Without prior navigation, still works but may encounter bot detection.

    Args:
        url: Full URL to resource (http:// or https://)
        output_filename: Filename to save as (no path). Saved to screenshot temp dir.

    Returns:
        {'path': '/tmp/.../file.js', 'size_bytes': 26703, 'content_type': '...', 'status': 200, 'url': '...'}

    Errors: Raises ToolError if browser not initialized, response status >= 400, or network failure.
    """
    if not url.startswith(("http://", "https://")):
        raise fastmcp.exceptions.ValidationError(
            "URL must start with http:// or https://"
        )

    _, _, page = await get_browser()

    if page is None:
        raise fastmcp.exceptions.ToolError(
            "Browser not initialized. Call navigate() first to establish browser session."
        )

    print(f"[download_resource] Downloading: {url}")

    # Use page context to maintain session/cookies
    response = await page.request.get(url)

    # Check response status
    if response.status >= 400:
        raise fastmcp.exceptions.ToolError(
            f"Download failed with status {response.status}: {url}"
        )

    # Get response body
    body = await response.body()

    # Sanitize filename (prevent path traversal, handle special chars)
    safe_filename = "".join(
        c if c.isalnum() or c in ".-_" else "_" for c in output_filename
    )
    if not safe_filename or safe_filename.startswith("."):
        safe_filename = "resource_" + safe_filename

    # Save to screenshot temp directory
    save_path = _screenshot_dir / safe_filename
    save_path.write_bytes(body)

    result = {
        "path": str(save_path),
        "size_bytes": len(body),
        "content_type": response.headers.get("content-type", "unknown"),
        "status": response.status,
        "url": url,
    }

    print(f"[download_resource] Downloaded {len(body)} bytes to {save_path}")

    return result


@mcp.tool(annotations=ToolAnnotations(title="Get ARIA Snapshot", readOnlyHint=True))
async def get_aria_snapshot(
    selector: str, include_urls: bool = False, timeout: int = 1000
) -> str:
    """Get semantic page structure as ARIA tree (91% compression vs raw HTML).

    Args:
        selector: CSS selector (e.g., 'body' for full page, '.wizard' for subset)
        include_urls: Include URL fields in output (default False saves ~25-30% tokens)
        timeout: Timeout in milliseconds (default 1000ms)

    Returns:
        YAML string with ARIA roles, names, and hierarchy (URLs filtered by default)
    """
    _, _, page = await get_browser()

    snapshot_yaml = await page.locator(selector).aria_snapshot(timeout=timeout)

    # Filter out /url: fields using YAML parser (preserves structure, saves ~25-30% tokens)
    if not include_urls:
        data = yaml.safe_load(snapshot_yaml)
        filtered_data = _remove_url_fields(data)
        snapshot_yaml = yaml.dump(
            filtered_data, default_flow_style=False, sort_keys=False
        )

    return snapshot_yaml


@mcp.tool(
    annotations=ToolAnnotations(title="Get Interactive Elements", readOnlyHint=True)
)
async def get_interactive_elements(
    selector_scope: str,
    text_contains: str | None,
    tag_filter: list[str] | None,
    limit: int | None,
    ctx: Context,
) -> list[InteractiveElement]:
    """Get clickable elements with optional filters for targeted extraction.

    Workflow: Use get_aria_snapshot() first for structure, then filter here for selectors.
    Example: get_interactive_elements(text_contains="Continue") → just Continue button

    Args:
        selector_scope: CSS selector to limit search (e.g., ".wizard" or "body")
        text_contains: Filter by text content (case-insensitive, None = no filter)
        tag_filter: Only specific tags (e.g., ["button", "a"], None = all tags)
        limit: Max results to return (None = unlimited)
        ctx: MCP context

    Returns:
        list[InteractiveElement]: Filtered interactive elements with selectors for clicking
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(f"Finding interactive elements in scope: {selector_scope}")

    # Prepare filter values for JS
    text_filter_lower = text_contains.lower() if text_contains else None
    tag_filter_upper = [tag.upper() for tag in tag_filter] if tag_filter else None

    elements = await page.evaluate(
        """(params) => {
        const { scopeSelector, textFilter, tagFilter, maxLimit } = params;
        const results = [];
        const scopeElement = document.querySelector(scopeSelector);
        if (!scopeElement) return results;

        const allElements = scopeElement.querySelectorAll('*');

        allElements.forEach((el, index) => {
            const style = window.getComputedStyle(el);
            const isVisible = style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            style.opacity !== '0';

            // Multiple detection methods
            const hasPointer = style.cursor === 'pointer';
            const isButton = el.tagName === 'BUTTON' || el.getAttribute('role') === 'button';
            const isLink = el.tagName === 'A' || el.getAttribute('role') === 'link';
            const isInput = ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName);

            const isInteractive = hasPointer || isButton || isLink || isInput;

            if (!isVisible || !isInteractive) return;

            // Apply tag filter
            if (tagFilter && !tagFilter.includes(el.tagName)) return;

            const text = el.textContent?.trim() || '';

            // Apply text filter
            if (textFilter && !text.toLowerCase().includes(textFilter)) return;

            // Apply limit
            if (maxLimit && results.length >= maxLimit) return;

            const selector = `${el.tagName.toLowerCase()}[data-automation-id="${index}"]`;
            el.setAttribute('data-automation-id', index);

            results.push({
                tag: el.tagName,
                text: text.substring(0, 100),
                selector: selector,
                cursor: style.cursor,
                href: el.href || null,
                classes: el.className || ''
            });
        });

        return results;
    }""",
        {
            "scopeSelector": selector_scope,
            "textFilter": text_filter_lower,
            "tagFilter": tag_filter_upper,
            "maxLimit": limit,
        },
    )

    await logger.info(f"Found {len(elements)} interactive elements (filtered)")
    return [InteractiveElement(**el) for el in elements]


@mcp.tool(
    annotations=ToolAnnotations(title="Get Focusable Elements", readOnlyHint=True)
)
async def get_focusable_elements(
    only_tabbable: bool, ctx: Context
) -> list[FocusableElement]:
    """Get keyboard-navigable elements sorted by tab order.

    Args:
        only_tabbable: True = Tab key only (tabindex >= 0)
                      False = includes programmatic focus (tabindex >= -1)
        ctx: MCP context

    Returns:
        list[FocusableElement]: Sorted by tab order, each with tag, text, selector, tab_index
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(f"Finding focusable elements (only_tabbable={only_tabbable})")

    min_tab_index = 0 if only_tabbable else -1

    elements = await page.evaluate(
        """(params) => {
        const { minTabIndex } = params;
        const results = [];
        const focusableSelectors = 'a[href], button, input, select, textarea, [tabindex]';
        const allElements = document.querySelectorAll(focusableSelectors);

        allElements.forEach((el, index) => {
            const style = window.getComputedStyle(el);
            const isVisible = style.display !== 'none' &&
                            style.visibility !== 'hidden';
            const tabIndex = el.tabIndex;
            const isFocusable = tabIndex >= minTabIndex;
            const isDisabled = el.disabled;

            if (isVisible && isFocusable && !isDisabled) {
                const text = el.textContent?.trim().substring(0, 100) || '';
                const selector = `${el.tagName.toLowerCase()}[data-focus-id="${index}"]`;
                el.setAttribute('data-focus-id', index);

                results.push({
                    tag: el.tagName,
                    text: text,
                    selector: selector,
                    tab_index: tabIndex,
                    is_tabbable: tabIndex >= 0,
                    classes: el.className || ''
                });
            }
        });

        return results.sort((a, b) => {
            if (a.tab_index === b.tab_index) return 0;
            if (a.tab_index === 0 && b.tab_index > 0) return 1;
            if (b.tab_index === 0 && a.tab_index > 0) return -1;
            return a.tab_index - b.tab_index;
        });
    }""",
        {"minTabIndex": min_tab_index},
    )

    await logger.info(f"Found {len(elements)} focusable elements")
    return [FocusableElement(**el) for el in elements]


@mcp.tool(
    annotations=ToolAnnotations(
        title="Click Element", destructiveHint=False, idempotentHint=False
    )
)
async def click(
    selector: str,
    ctx: Context,
    wait_for_network: bool = False,
    network_timeout: int = 10000,
):
    """Click element. Auto-waits for element to be visible, enabled, and stable.

    Args:
        selector: CSS selector from get_interactive_elements() or get_focusable_elements()
        ctx: MCP context
        wait_for_network: If True, wait for network idle after clicking (default False)
        network_timeout: Timeout in ms for network idle wait (default 10000ms)

    Note: Set wait_for_network=True for navigation clicks that trigger dynamic content.
          For non-navigation clicks (modals, dropdowns), leave False for faster execution.
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(
        f"Clicking element: {selector}"
        + (" (with network wait)" if wait_for_network else "")
    )

    await page.click(selector)

    if wait_for_network:
        await logger.info(f"Waiting for network idle (timeout={network_timeout}ms)")
        await page.wait_for_load_state("networkidle", timeout=network_timeout)
        await logger.info("Network idle")

    await logger.info("Click successful")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Wait for Network Idle", readOnlyHint=True, idempotentHint=True
    )
)
async def wait_for_network_idle(ctx: Context, timeout: int = 10000):
    """Wait for network activity to settle after clicks or dynamic content loads.

    Use after click() if page triggers AJAX/fetch requests. Default: 10000ms (10 sec).

    Args:
        ctx: MCP context
        timeout: Timeout in milliseconds (default 10000ms)
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(f"Waiting for network idle (timeout={timeout}ms)")

    await page.wait_for_load_state("networkidle", timeout=timeout)

    await logger.info("Network idle")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Press Keyboard Key", destructiveHint=False, idempotentHint=False
    )
)
async def press_key(key: str, ctx: Context):
    """Press a keyboard key or key combination.

    Args:
        key: Key name or combination. Common keys:
            - Single keys: 'Escape', 'Enter', 'Tab', 'Backspace', 'Delete', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'
            - Function keys: 'F1' through 'F12'
            - Key combinations: 'Control+A' (Ctrl+A), 'Meta+V' (Cmd+V on Mac), 'Control+Shift+T', 'Alt+F4'
            - Modifiers: 'Control', 'Shift', 'Alt', 'Meta'
        ctx: MCP context

    Examples:
        - press_key('Escape') - Close modals
        - press_key('Enter') - Submit forms
        - press_key('Tab') - Navigate between fields
        - press_key('Control+A') - Select all (Ctrl+A on Windows/Linux)
        - press_key('Meta+V') - Paste (Cmd+V on Mac)

    Note: Key names are case-sensitive. Use logical key names (e.g., 'KeyA' for letter A, 'Digit1' for number 1).
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(f"Pressing key: {key}")

    await page.keyboard.press(key)

    await logger.info("Key press successful")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Type Text", destructiveHint=False, idempotentHint=False
    )
)
async def type_text(text: str, ctx: Context, delay_ms: int = 0):
    """Type text character by character with optional delay between keystrokes.

    Args:
        text: Text to type
        ctx: MCP context
        delay_ms: Optional delay between keystrokes in milliseconds (default 0)

    Note: For simple form filling, prefer using page.fill() via other tools as it's faster.
          Use this tool when you need to simulate human typing or trigger per-character keyboard events.

    Example:
        - type_text('Hello, world!') - Type text instantly
        - type_text('search query', delay_ms=50) - Type with 50ms delay between chars
    """
    logger = DualLogger(ctx)
    _, _, page = await get_browser()

    await logger.info(
        f'Typing text: "{text}"' + (f" with {delay_ms}ms delay" if delay_ms > 0 else "")
    )

    await page.keyboard.type(text, delay=delay_ms)

    await logger.info("Text typing successful")


def main():
    """Entry point for uvx installation."""
    print("Starting Browser Automation MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
