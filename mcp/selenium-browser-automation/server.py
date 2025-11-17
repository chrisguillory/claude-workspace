#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "fastmcp>=2.12.5",
#   "selenium",
#   "httpx",
#   "pyyaml",
# ]
# ///
#
# Selenium Browser Automation MCP Server
#
# Architecture: Runs locally via uv --script (not Docker) for visible browser monitoring.
# Uses Selenium with CDP stealth injection to bypass Cloudflare bot detection.
#
# Setup:
#   claude mcp add --transport stdio selenium-browser-automation -- uv run --script "$(git rev-parse --show-toplevel)/mcp/selenium-browser-automation/server.py"

from __future__ import annotations

# Standard Library
import asyncio
import base64
import json
import sys
import tempfile
import time
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

# Third-Party Libraries
import fastmcp.exceptions
import httpx
import yaml
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Local imports
from src.chrome_profiles import list_all_profiles, get_chrome_base_path
from src.models import ChromeProfilesResult


class PrintLogger:
    """Simple logger that logs to stderr (MCP servers must not write to stdout)."""

    def __init__(self, ctx: Context | None = None):
        """Initialize logger. Context is ignored."""
        pass

    async def info(self, message: str):
        """Log info message to stderr."""
        print(f"[selenium-browser-automation] {message}", file=sys.stderr)


"""
Architecture: All tools operate on BrowserService (service pattern with shared state).
Workflow: navigate(url) → get_page_content() → click() → screenshot()

Critical: CDP stealth injection bypasses Cloudflare where Playwright fails.
Browser Context: All requests share session/cookies - bypasses bot detection.
Temp Files: Auto-cleanup on shutdown via state.temp_dir.

Enhanced Features (v2):
✅ ARIA Snapshot - JavaScript-based accessible name computation per WAI-ARIA spec
✅ Network Idle - CDP Network domain monitoring with 500ms idle threshold
✅ Full-Page Screenshots - CDP Page.captureScreenshot with captureBeyondViewport
✅ Resource Capture - CDP Network interception with HAR export support
"""


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


class BrowserState:
    """Container for all browser state - initialized once at startup, never Optional."""

    @classmethod
    async def create(cls) -> typing.Self:
        """Factory method to create and initialize browser state."""
        # Initialize temp directories
        temp_dir = tempfile.TemporaryDirectory()
        screenshot_dir = Path(temp_dir.name)

        capture_temp_dir = tempfile.TemporaryDirectory()
        capture_dir = Path(capture_temp_dir.name)

        print("[BrowserState] Temp directories initialized", file=sys.stderr)
        print(f"  Screenshots: {screenshot_dir}", file=sys.stderr)
        print(f"  Captures: {capture_dir}", file=sys.stderr)

        return cls(
            driver=None,
            current_profile=None,
            temp_dir=temp_dir,
            screenshot_dir=screenshot_dir,
            capture_temp_dir=capture_temp_dir,
            capture_dir=capture_dir,
            capture_counter=0,
        )

    def __init__(
        self,
        driver: webdriver.Chrome
        | None,  # None = lazy initialization (created on first use)
        current_profile: str
        | None,  # None = no profile selected (temporary profile mode)
        temp_dir: tempfile.TemporaryDirectory,
        screenshot_dir: Path,
        capture_temp_dir: tempfile.TemporaryDirectory,
        capture_dir: Path,
        capture_counter: int,
    ) -> None:
        self.driver = driver  # Lazy-initialized: None until first navigation
        self.current_profile = current_profile  # None when using temporary profile
        self.temp_dir = temp_dir
        self.screenshot_dir = screenshot_dir
        self.capture_temp_dir = capture_temp_dir
        self.capture_dir = capture_dir
        self.capture_counter = capture_counter


class LoggerProtocol(typing.Protocol):
    """Protocol for logger - allows service to be MCP-agnostic."""

    async def info(self, message: str) -> None: ...


class BrowserService:
    """Browser automation service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: BrowserState) -> None:
        self.state = state  # Non-Optional - guaranteed by constructor

    async def close_browser(self) -> None:
        """Tear down browser and reset profile state."""
        if self.state.driver:
            await asyncio.to_thread(self.state.driver.quit)
            self.state.driver = None
        self.state.current_profile = None

    async def get_browser(self, profile: str | None = None) -> webdriver.Chrome:
        """Initialize and return browser session (lazy singleton pattern).

        Args:
            profile: Chrome profile directory (e.g., "Default", "Profile 1")
                     If None, uses fresh temporary profile (no saved logins)

        Returns:
            WebDriver instance

        Raises:
            ValidationError: If specified profile doesn't exist

        Note: If profile changes from current session, browser will be recreated.
              Chrome locks profiles, so you can't have the same profile open in both
              regular Chrome and this automation at the same time.
        """
        # If profile changed, need fresh browser
        if self.state.driver is not None and profile != self.state.current_profile:
            await self.close_browser()

        if self.state.driver is not None:
            return self.state.driver

        # CRITICAL: Stealth configuration to bypass Cloudflare bot detection
        opts = Options()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--window-size=1920,1080")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_experimental_option(
            "prefs",
            {
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False,
            },
        )

        # Add profile if specified
        if profile:
            chrome_base = get_chrome_base_path()
            profile_path = chrome_base / profile

            if not profile_path.exists():
                raise fastmcp.exceptions.ValidationError(
                    f"Chrome profile not found: {profile}\n"
                    f"Use list_chrome_profiles() to see available profiles"
                )

            opts.add_argument(f"--user-data-dir={chrome_base}")
            opts.add_argument(f"--profile-directory={profile}")
            print(f"[browser] Using Chrome profile: {profile}", file=sys.stderr)

        # Initialize driver in thread pool (blocking operation)
        self.state.driver = await asyncio.to_thread(webdriver.Chrome, options=opts)
        self.state.current_profile = profile

        # CRITICAL: CDP injection AFTER driver creation but BEFORE first navigation
        # This is what makes Selenium bypass Cloudflare where Playwright fails
        await asyncio.to_thread(
            self.state.driver.execute_cdp_cmd,
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    window.navigator.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """
            },
        )

        return self.state.driver


def register_tools(service: BrowserService) -> None:
    """Register service methods as MCP tools via closures."""

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
        fresh_browser: bool = False,
        profile: str | None = None,
        ctx: Context | None = None,
    ) -> NavigationResult:
        """Navigate to URL using Selenium with CDP stealth injection.

        Args:
            url: Full URL (http:// or https://)
            fresh_browser: If True, tear down and recreate browser for clean session (no cache/cookies)
            profile: Chrome profile directory (e.g., "Default", "Profile 1"). If None, uses temporary profile.
            ctx: MCP context

        Returns:
            NavigationResult with current_url and title

        Note: Resource capture not implemented in Selenium version (complex, requires network interception)
        """
        if not url.startswith(("http://", "https://")):
            raise fastmcp.exceptions.ValidationError(
                "URL must start with http:// or https://"
            )

        print(
            f"[navigate] Navigating to {url}"
            + (" (fresh browser)" if fresh_browser else ""),
            file=sys.stderr,
        )

        if fresh_browser:
            await service.close_browser()

        driver = await service.get_browser(profile=profile)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        print(
            f"[navigate] Successfully navigated to {driver.current_url}",
            file=sys.stderr,
        )

        return NavigationResult(current_url=driver.current_url, title=driver.title)

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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(
            f"Extracting as {format}"
            + (f' with selector "{selector}"' if selector else "")
        )

        if selector:
            # Extract specific elements
            elements = await asyncio.to_thread(
                driver.find_elements, By.CSS_SELECTOR, selector
            )
            count = len(elements)

            actual_limit = min(count, limit) if limit is not None else count

            html_parts = []
            for i in range(actual_limit):
                html = await asyncio.to_thread(
                    driver.execute_script, "return arguments[0].outerHTML;", elements[i]
                )
                html_parts.append(html)

            await logger.info(f"Matched {len(html_parts)} elements")
            return "\n".join(html_parts)

        # Full page extraction
        if format == "text":
            body = await asyncio.to_thread(driver.find_element, By.TAG_NAME, "body")
            content = await asyncio.to_thread(lambda: body.text)
            return content
        elif format == "html":
            content = await asyncio.to_thread(lambda: driver.page_source)
            return content
        elif format == "markdown":
            raise fastmcp.exceptions.ValidationError(
                "Markdown format not yet implemented"
            )
        else:
            raise fastmcp.exceptions.ValidationError(f"Unknown format: {format}")

    @mcp.tool(annotations=ToolAnnotations(title="Take Screenshot", readOnlyHint=True))
    async def screenshot(filename: str, ctx: Context, full_page: bool = False) -> str:
        """Take screenshot (viewport or full-page). Saves to temp directory with auto-cleanup on exit.

        Args:
            filename: Name for screenshot file (e.g., "homepage.png")
            ctx: MCP context
            full_page: If True, capture entire scrollable page using CDP (Chrome only). Default False.

        Returns:
            Absolute path to temp file (use Read tool to view image)

        Note: full_page=True uses CDP Page.captureScreenshot with captureBeyondViewport.
              Falls back to viewport-only if CDP fails.
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        screenshot_path = service.state.screenshot_dir / filename

        if full_page:
            await logger.info(f"Taking full-page screenshot: {filename}")
            try:
                # Use CDP to capture full page
                result = await asyncio.to_thread(
                    driver.execute_cdp_cmd,
                    "Page.captureScreenshot",
                    {"captureBeyondViewport": True},
                )

                if "data" in result:
                    # Result contains base64-encoded PNG
                    screenshot_data = base64.b64decode(result["data"])
                    screenshot_path.write_bytes(screenshot_data)
                    await logger.info(
                        f"Full-page screenshot saved to {screenshot_path}"
                    )
                    return str(screenshot_path)
                else:
                    await logger.info(
                        "CDP capture returned no data, falling back to viewport"
                    )
            except Exception as e:
                await logger.info(f"CDP capture failed ({e}), falling back to viewport")

        # Viewport screenshot (default or fallback)
        await logger.info(f"Taking viewport screenshot: {filename}")
        await asyncio.to_thread(driver.save_screenshot, str(screenshot_path))
        await logger.info(f"Screenshot saved to {screenshot_path}")
        return str(screenshot_path)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Download Specific Resource", readOnlyHint=False, idempotentHint=False
        )
    )
    async def download_resource(url: str, output_filename: str) -> dict:
        """Download specific resource using httpx with cookies from current browser session.

        Uses driver.get_cookies() to maintain session from prior navigation.
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

        driver = await service.get_browser()

        if driver is None:
            raise fastmcp.exceptions.ToolError(
                "Browser not initialized. Call navigate() first to establish browser session."
            )

        print(f"[download_resource] Downloading: {url}", file=sys.stderr)

        # Get cookies from driver to maintain session
        selenium_cookies = await asyncio.to_thread(driver.get_cookies)

        # Convert Selenium cookies to httpx format
        cookies = {cookie["name"]: cookie["value"] for cookie in selenium_cookies}

        # Download using httpx with session cookies
        async with httpx.AsyncClient() as client:
            response = await client.get(url, cookies=cookies, follow_redirects=True)

        # Check response status
        if response.status_code >= 400:
            raise fastmcp.exceptions.ToolError(
                f"Download failed with status {response.status_code}: {url}"
            )

        # Get response body
        body = response.content

        # Sanitize filename (prevent path traversal, handle special chars)
        safe_filename = "".join(
            c if c.isalnum() or c in ".-_" else "_" for c in output_filename
        )
        if not safe_filename or safe_filename.startswith("."):
            safe_filename = "resource_" + safe_filename

        # Save to screenshot temp directory
        save_path = service.state.screenshot_dir / safe_filename
        save_path.write_bytes(body)

        result = {
            "path": str(save_path),
            "size_bytes": len(body),
            "content_type": response.headers.get("content-type", "unknown"),
            "status": response.status_code,
            "url": url,
        }

        print(
            f"[download_resource] Downloaded {len(body)} bytes to {save_path}",
            file=sys.stderr,
        )

        return result

    @mcp.tool(annotations=ToolAnnotations(title="Get ARIA Snapshot", readOnlyHint=True))
    async def get_aria_snapshot(
        selector: str, include_urls: bool = False, timeout: int = 1000
    ) -> str:
        """Get semantic page structure as ARIA tree via JavaScript accessible name computation.

        Uses JavaScript to compute accessible names per WAI-ARIA spec and build accessibility tree.
        91% compression vs raw HTML.

        Args:
            selector: CSS selector (e.g., 'body' for full page, '.wizard' for subset)
            include_urls: Include URL fields in output (default False saves ~25-30% tokens)
            timeout: Timeout in milliseconds (unused - kept for compatibility)

        Returns:
            YAML string with ARIA roles, names, and hierarchy (URLs filtered by default)
        """
        driver = await service.get_browser()

        # JavaScript for accessible name computation per WAI-ARIA spec
        aria_script = """
        function getAccessibilitySnapshot(rootSelector, includeUrls) {
            const root = document.querySelector(rootSelector);
            if (!root) return null;

            // Skip non-rendered elements
            const SKIP_TAGS = ['SCRIPT', 'STYLE', 'META', 'LINK', 'NOSCRIPT'];

            function isVisible(el) {
                const style = window.getComputedStyle(el);
                return style.display !== 'none' &&
                       style.visibility !== 'hidden' &&
                       style.opacity !== '0';
            }

            // Accessible name computation per WAI-ARIA spec
            function computeAccessibleName(el) {
                // Step 1: aria-label
                if (el.getAttribute('aria-label')) {
                    return el.getAttribute('aria-label').trim();
                }

                // Step 2: aria-labelledby
                if (el.getAttribute('aria-labelledby')) {
                    const ids = el.getAttribute('aria-labelledby').split(/\\s+/);
                    return ids
                        .map(id => document.getElementById(id)?.textContent || '')
                        .filter(Boolean)
                        .join(' ');
                }

                // Step 3: Label element association
                if (el.id) {
                    const label = document.querySelector(`label[for="${el.id}"]`);
                    if (label) return label.textContent.trim();
                }

                // Step 4: Implicit label (form control inside label)
                if (el.closest('label')) {
                    return el.closest('label').textContent.trim();
                }

                // Step 5: Element content for links, buttons, headings
                const tagName = el.tagName.toLowerCase();
                if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
                    return el.textContent.trim();
                }

                // Step 6: Title attribute
                if (el.getAttribute('title')) {
                    return el.getAttribute('title');
                }

                // Step 7: Alt text for images
                if (tagName === 'img') {
                    return el.getAttribute('alt') || '';
                }

                // Step 8: Placeholder for inputs
                if (['input', 'textarea'].includes(tagName)) {
                    return el.placeholder || el.value || '';
                }

                return '';
            }

            // Implicit role mapping per HTML AAM spec
            function getImplicitRole(el) {
                const tagName = el.tagName.toLowerCase();
                const type = el.getAttribute('type')?.toLowerCase();

                const implicitRoles = {
                    'a': 'link',
                    'button': 'button',
                    'h1': 'heading', 'h2': 'heading', 'h3': 'heading',
                    'h4': 'heading', 'h5': 'heading', 'h6': 'heading',
                    'header': 'banner',
                    'footer': 'contentinfo',
                    'nav': 'navigation',
                    'main': 'main',
                    'article': 'article',
                    'section': 'region',
                    'aside': 'complementary',
                    'form': 'form',
                    'p': 'paragraph',
                    'input': type === 'checkbox' ? 'checkbox' : type === 'radio' ? 'radio' : 'textbox',
                    'textarea': 'textbox',
                    'select': 'combobox',
                    'ul': 'list',
                    'ol': 'list',
                    'li': 'listitem',
                    'table': 'table',
                    'tr': 'row',
                    'td': 'cell',
                    'th': 'columnheader',
                    'img': 'img',
                    'strong': 'strong',
                    'em': 'emphasis',
                    'code': 'code'
                };

                return implicitRoles[tagName] || 'generic';
            }

            // Walk tree and build hierarchical snapshot (includes text nodes!)
            function walkTree(el, depth = 0) {
                if (depth > 50) return null; // Prevent infinite recursion

                // Handle text nodes
                if (el.nodeType === Node.TEXT_NODE) {
                    const text = el.textContent.trim();
                    if (text) {
                        return { type: 'text', content: text };
                    }
                    return null;
                }

                // Skip non-element nodes
                if (el.nodeType !== Node.ELEMENT_NODE) return null;

                // Skip non-rendered elements
                if (SKIP_TAGS.includes(el.tagName)) return null;

                // Skip hidden elements
                if (!isVisible(el)) return null;

                const role = el.getAttribute('role') || getImplicitRole(el);
                const name = computeAccessibleName(el);

                const node = { role: role };

                // Add name if available
                if (name) {
                    node.name = name;
                }

                // Add description if available
                if (el.getAttribute('aria-description')) {
                    node.description = el.getAttribute('aria-description');
                }

                // Add level for headings
                if (role === 'heading') {
                    const match = el.tagName.match(/h([1-6])/i);
                    if (match) {
                        node.level = parseInt(match[1]);
                    }
                }

                // Add checked state for checkboxes/radios
                if (['checkbox', 'radio', 'switch'].includes(role)) {
                    if (el.hasAttribute('aria-checked')) {
                        node.checked = el.getAttribute('aria-checked') === 'true';
                    } else if (el.tagName === 'INPUT') {
                        node.checked = el.checked;
                    }
                }

                // Add disabled state
                if (el.hasAttribute('aria-disabled')) {
                    node.disabled = el.getAttribute('aria-disabled') === 'true';
                } else if (['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) {
                    node.disabled = el.disabled;
                }

                // Add URL for links if requested
                if (includeUrls && role === 'link' && el.href) {
                    node.url = el.href;
                }

                // Process child NODES (not just elements - includes text!)
                const children = [];
                for (const child of el.childNodes) {
                    const childNode = walkTree(child, depth + 1);
                    if (childNode) {
                        children.push(childNode);
                    }
                }

                // Add children array if not empty
                if (children.length > 0) {
                    node.children = children;
                }

                return node;
            }

            return walkTree(root);
        }

        return getAccessibilitySnapshot(arguments[0], arguments[1]);
        """

        # Execute script and get snapshot data
        snapshot_data = await asyncio.to_thread(
            driver.execute_script, aria_script, selector, include_urls
        )

        # Convert to Playwright-compatible YAML format using custom serializer
        def serialize_aria_snapshot(node, indent=0):
            """Custom serializer matching Playwright's ARIA snapshot YAML format."""
            if node is None:
                return ""

            lines = []
            prefix = " " * indent + "- "

            # Handle text nodes
            if node.get("type") == "text":
                content = node.get("content", "").replace("\n", " ")
                lines.append(f"{prefix}text: {content}")
                return "\n".join(lines)

            # Handle element nodes
            role = node.get("role", "generic")
            name = node.get("name", "")
            children = node.get("children", [])

            # Build node header in Playwright format: role "name" [attrs]:
            header = f"{prefix}{role}"

            if name:
                # Escape quotes in name
                escaped_name = name.replace('"', '\\"')
                header += f' "{escaped_name}"'

            # Add attributes in brackets
            attrs = []
            if "level" in node:
                attrs.append(f"level={node['level']}")
            if node.get("checked"):
                attrs.append("checked")
            if node.get("disabled"):
                attrs.append("disabled")

            if attrs:
                header += f" [{', '.join(attrs)}]"

            # Add colon if has children
            if children:
                header += ":"

            lines.append(header)

            # Process children with increased indentation
            if children:
                for child in children:
                    child_output = serialize_aria_snapshot(child, indent + 2)
                    if child_output:
                        lines.append(child_output)

            return "\n".join(lines)

        return serialize_aria_snapshot(snapshot_data)

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

        Workflow: Use this tool to find elements, then use click() with returned selectors.
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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Finding interactive elements in scope: {selector_scope}")

        # Prepare filter values for JS
        text_filter_lower = text_contains.lower() if text_contains else None
        tag_filter_upper = [tag.upper() for tag in tag_filter] if tag_filter else None

        elements = await asyncio.to_thread(
            driver.execute_script,
            """
            const params = arguments[0];
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
            """,
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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Finding focusable elements (only_tabbable={only_tabbable})")

        min_tab_index = 0 if only_tabbable else -1

        elements = await asyncio.to_thread(
            driver.execute_script,
            """
            const params = arguments[0];
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
            """,
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
        """Click element. Auto-waits for element to be visible and clickable.

        Args:
            selector: CSS selector from get_interactive_elements() or get_focusable_elements()
            ctx: MCP context
            wait_for_network: If True, adds fixed delay after clicking (default False)
            network_timeout: Delay in ms after click (default 10000ms)

        Note: Selenium lacks native network idle detection. wait_for_network adds fixed delay.
              Set wait_for_network=True for navigation clicks that trigger dynamic content.
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(
            f"Clicking element: {selector}"
            + (" (with delay)" if wait_for_network else "")
        )

        # Wait for element to be clickable (up to 10 seconds)
        element = await asyncio.to_thread(
            WebDriverWait(driver, 10).until,
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector)),
        )

        # Click element
        await asyncio.to_thread(element.click)

        if wait_for_network:
            delay_sec = network_timeout / 1000
            await logger.info(f"Waiting {delay_sec}s for content to load")
            await asyncio.sleep(delay_sec)
            await logger.info("Delay complete")

        await logger.info("Click successful")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Wait for Network Idle", readOnlyHint=True, idempotentHint=True
        )
    )
    async def wait_for_network_idle(ctx: Context, timeout: int = 10000):
        """Wait for network activity to settle after clicks or dynamic content loads.

        Uses JavaScript instrumentation to monitor Fetch and XMLHttpRequest activity.
        Waits for no active requests and 500ms of idle time.

        Args:
            ctx: MCP context
            timeout: Timeout in milliseconds (default 10000ms)

        Note: Uses JavaScript instrumentation of Fetch/XHR APIs to track network activity.
              Waits for no active requests + 500ms idle threshold.
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        # Step 1: Inject monitoring script to instrument Fetch and XHR
        setup_script = """
        window.__networkMonitor = {
            activeRequests: 0,
            lastRequestTime: null,

            increment() {
                this.activeRequests++;
                this.lastRequestTime = Date.now();
            },

            decrement() {
                this.activeRequests = Math.max(0, this.activeRequests - 1);
            }
        };

        // Instrument Fetch API
        const origFetch = window.fetch;
        window.fetch = function(...args) {
            window.__networkMonitor.increment();
            return origFetch.apply(this, args)
                .then(r => { window.__networkMonitor.decrement(); return r; })
                .catch(e => { window.__networkMonitor.decrement(); throw e; });
        };

        // Instrument XMLHttpRequest
        const origOpen = XMLHttpRequest.prototype.open;
        const origSend = XMLHttpRequest.prototype.send;

        XMLHttpRequest.prototype.open = function() {
            window.__networkMonitor.increment();
            return origOpen.apply(this, arguments);
        };

        XMLHttpRequest.prototype.send = function() {
            this.addEventListener('loadend', () => window.__networkMonitor.decrement());
            return origSend.apply(this, arguments);
        };
        """

        await asyncio.to_thread(driver.execute_script, setup_script)
        await logger.info("Network monitor injected")

        # Step 2: Poll for idle state (500ms threshold)
        check_script = """
        return {
            activeRequests: window.__networkMonitor?.activeRequests || 0,
            lastRequestTime: window.__networkMonitor?.lastRequestTime,
            currentTime: Date.now()
        };
        """

        start_time = time.time()
        idle_threshold_ms = 500
        timeout_s = timeout / 1000

        while time.time() - start_time < timeout_s:
            status = await asyncio.to_thread(driver.execute_script, check_script)

            if status["activeRequests"] == 0:
                if status["lastRequestTime"] is None:
                    # No requests made yet - check elapsed time since monitoring started
                    elapsed = time.time() - start_time
                    if elapsed >= idle_threshold_ms / 1000:
                        await logger.info("Network idle (no requests made)")
                        return
                else:
                    # Check time since last request completed
                    elapsed_since_last = (
                        status["currentTime"] - status["lastRequestTime"]
                    )
                    if elapsed_since_last >= idle_threshold_ms:
                        await logger.info(
                            f"Network idle after {elapsed_since_last / 1000:.2f}s"
                        )
                        return

            await asyncio.sleep(0.05)  # Poll every 50ms

        await logger.info(f"Network idle timeout after {timeout_s}s")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Press Keyboard Key", destructiveHint=False, idempotentHint=False
        )
    )
    async def press_key(key: str, ctx: Context):
        """Press a keyboard key or key combination.

        Args:
            key: Key name or combination. Common keys:
                - Single keys: 'ESCAPE', 'ENTER', 'TAB', 'BACKSPACE', 'DELETE', 'ARROW_UP', 'ARROW_DOWN', 'ARROW_LEFT', 'ARROW_RIGHT'
                - Special keys: 'F1' through 'F12', 'HOME', 'END', 'PAGE_UP', 'PAGE_DOWN'
                - Modifiers: Use + for combinations like 'CONTROL+A', 'META+V'
            ctx: MCP context

        Examples:
            - press_key('ESCAPE') - Close modals
            - press_key('ENTER') - Submit forms
            - press_key('TAB') - Navigate between fields
            - press_key('CONTROL+A') - Select all
            - press_key('META+V') - Paste (Cmd+V on Mac)

        Note: Key names use Selenium's Keys enum (uppercase with underscores).
              For combinations, use + (e.g., 'CONTROL+A').
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Pressing key: {key}")

        # Map common key names to Selenium Keys
        # Handle key combinations (e.g., "CONTROL+A")
        if "+" in key:
            parts = key.split("+")
            keys_combo = []
            for part in parts:
                key_attr = part.upper().replace(" ", "_")
                if hasattr(Keys, key_attr):
                    keys_combo.append(getattr(Keys, key_attr))
                else:
                    keys_combo.append(part)

            # Send key combination
            body = await asyncio.to_thread(driver.find_element, By.TAG_NAME, "body")
            await asyncio.to_thread(body.send_keys, *keys_combo)
        else:
            # Single key
            key_attr = key.upper().replace(" ", "_")
            if hasattr(Keys, key_attr):
                key_value = getattr(Keys, key_attr)
            else:
                key_value = key

            body = await asyncio.to_thread(driver.find_element, By.TAG_NAME, "body")
            await asyncio.to_thread(body.send_keys, key_value)

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

        Note: For simple form filling, prefer using element.send_keys() directly as it's faster.
              Use this tool when you need to simulate human typing or trigger per-character keyboard events.

        Example:
            - type_text('Hello, world!') - Type text instantly
            - type_text('search query', delay_ms=50) - Type with 50ms delay between chars
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(
            f'Typing text: "{text}"'
            + (f" with {delay_ms}ms delay" if delay_ms > 0 else "")
        )

        active_element = await asyncio.to_thread(
            lambda: driver.switch_to.active_element
        )

        if delay_ms > 0:
            # Type with delay between characters
            for char in text:
                await asyncio.to_thread(active_element.send_keys, char)
                await asyncio.sleep(delay_ms / 1000)
        else:
            # Type all at once
            await asyncio.to_thread(active_element.send_keys, text)

        await logger.info("Text typing successful")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List Chrome Profiles", readOnlyHint=True, idempotentHint=True
        )
    )
    async def list_chrome_profiles(
        verbose: bool = False, ctx: Context | None = None
    ) -> ChromeProfilesResult:
        """List all Chrome profiles with metadata.

        Args:
            verbose: If True, include all metadata fields.
                     If False (default), show only essential fields.
            ctx: MCP context (optional, for logging)

        Returns:
            ChromeProfilesResult with profiles list, count, and default profile

        Example (default):
            Returns essential fields: name, email, profile_dir, etc.

        Example (verbose):
            Returns all fields including avatar settings, creation time, etc.
        """
        if ctx:
            logger = PrintLogger(ctx)
            await logger.info(f"Listing Chrome profiles (verbose={verbose})")

        # Call module function
        result = await asyncio.to_thread(list_all_profiles, verbose=verbose)

        if ctx:
            await logger.info(f"Found {result.total_count} profiles")

        return result


@asynccontextmanager
async def lifespan(server_instance: FastMCP) -> typing.AsyncIterator[None]:
    """Manage browser lifecycle - initialization before requests, cleanup after shutdown."""
    state = await BrowserState.create()
    service = BrowserService(state)
    register_tools(service)

    print("✓ Browser service initialized", file=sys.stderr)
    print(f"  Screenshot directory: {state.screenshot_dir}", file=sys.stderr)
    print(f"  Capture directory: {state.capture_dir}", file=sys.stderr)

    yield

    # SHUTDOWN: Cleanup after all requests complete
    if state.driver:
        await asyncio.to_thread(state.driver.quit)
    state.temp_dir.cleanup()
    state.capture_temp_dir.cleanup()
    print("✓ Server cleanup complete", file=sys.stderr)


mcp = FastMCP("selenium-browser-automation", lifespan=lifespan)


if __name__ == "__main__":
    print("Starting Selenium Browser Automation MCP server", file=sys.stderr)
    print(
        "Note: This server uses CDP stealth injection to bypass bot detection",
        file=sys.stderr,
    )
    mcp.run()
