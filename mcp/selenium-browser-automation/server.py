#!/usr/bin/env python3
"""
Selenium Browser Automation MCP Server

CDP stealth injection to bypass Cloudflare bot detection.

Install:
    uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser-automation server

Architecture: Runs locally (not Docker) for visible browser monitoring.
Uses Selenium with CDP stealth injection where Playwright fails.
"""

from __future__ import annotations

# Standard Library
import asyncio
import base64
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
import typing
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

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
from selenium.common.exceptions import WebDriverException

# Local imports
from src.chrome_profiles import list_all_profiles, get_chrome_base_path
from src.models import (
    ChromeProfilesResult,
    CoreWebVitals,
    FCPMetric,
    LCPMetric,
    TTFBMetric,
    TTFBPhases,
    CLSMetric,
    LayoutShiftEntry,
    LayoutShiftSource,
    INPMetric,
    INPDetails,
    NetworkCapture,
    NetworkRequest,
    RequestTiming,
    JavaScriptResult,
)


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
Workflow: navigate(url) → get_page_text() → click() → screenshot()

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


class HARExportResult(BaseModel):
    """Result of HAR export operation."""
    path: str
    entry_count: int
    size_bytes: int
    has_errors: bool = False
    errors: list[str] = []


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


class SmartExtractionInfo(BaseModel):
    """Metadata about smart extraction decisions. Only present for selector='auto'."""

    fallback_used: bool  # True if no suitable main/article found, fell back to body
    body_character_count: int  # Total body chars for coverage: character_count/body_character_count


class PageTextResult(BaseModel):
    """Result of text extraction operation."""

    # Core content (always present)
    title: str
    url: str
    text: str
    character_count: int
    source_element: str  # What was extracted: "main", "article", "body", or CSS selector

    # Smart extraction transparency (only present for selector='auto')
    smart_info: SmartExtractionInfo | None = None


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
        # Network capture state (CDP-based on-demand capture)
        self.network_capture_enabled = False
        self.network_capture_start_time: float | None = None
        self.network_events: list[dict] = []  # Collected CDP events
        self.network_resource_filter: list[str] | None = None  # Optional resource type filter
        # Proxy configuration for authenticated proxies
        self.proxy_config: dict[str, str] | None = None  # {host, port, username, password}
        self.mitmproxy_process: subprocess.Popen | None = None  # Local mitmproxy for upstream auth


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
        # Reset network capture state
        self.state.network_capture_enabled = False
        self.state.network_capture_start_time = None
        self.state.network_events = []
        self.state.network_resource_filter = None

    async def get_browser(self, profile: str | None = None, enable_har_capture: bool = False) -> webdriver.Chrome:
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
        # Performance logging for HAR export (opt-in due to overhead)
        # When enabled, Chrome continuously buffers Network.* events which adds
        # CPU, memory, and data transfer overhead even when export_har() isn't called.
        if enable_har_capture:
            opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
            opts.add_experimental_option("perfLoggingPrefs", {"enableNetwork": True})

        # Apply proxy configuration if mitmproxy is running
        # mitmproxy handles upstream authentication, Chrome just connects to local proxy
        if self.state.proxy_config and self.state.mitmproxy_process:
            opts.add_argument('--proxy-server=http://127.0.0.1:8080')
            opts.add_argument('--ignore-certificate-errors')  # mitmproxy uses self-signed certs
            print(f"[browser] Using local mitmproxy -> {self.state.proxy_config['host']}:{self.state.proxy_config['port']}", file=sys.stderr)

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
        enable_har_capture: bool = False,
        ctx: Context | None = None,
    ) -> NavigationResult:
        """Load a URL and establish browser session. Entry point for all browser automation.

        After navigation completes, call get_aria_snapshot('body') to understand page structure
        before interacting with elements.

        Args:
            url: Full URL (http:// or https://)
            fresh_browser: If True, creates clean session (no cache/cookies)
            profile: Chrome profile directory for authenticated sessions (e.g., "Default")
            enable_har_capture: If True, enables performance logging for HAR export.
                               Requires fresh_browser=True (adds overhead, must be set at browser init).

        Returns:
            NavigationResult with current_url and title

        Next steps:
            1. get_aria_snapshot('body') - understand page structure
            2. get_interactive_elements() - find specific elements to click
            3. click(selector) - interact with elements

        For performance investigation:
            Call start_network_capture() BEFORE navigate() to measure page load timing.
        """
        valid_prefixes = ("http://", "https://", "file://", "about:", "data:")
        if not url.startswith(valid_prefixes):
            raise fastmcp.exceptions.ValidationError(
                "URL must start with http://, https://, file://, about:, or data:"
            )

        if enable_har_capture and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                "enable_har_capture requires fresh_browser=True (performance logging must be set at browser init)"
            )

        print(
            f"[navigate] Navigating to {url}"
            + (" (fresh browser)" if fresh_browser else "")
            + (" (HAR capture enabled)" if enable_har_capture else ""),
            file=sys.stderr,
        )

        if fresh_browser:
            await service.close_browser()

        driver = await service.get_browser(profile=profile, enable_har_capture=enable_har_capture)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        print(
            f"[navigate] Successfully navigated to {driver.current_url}",
            file=sys.stderr,
        )

        return NavigationResult(current_url=driver.current_url, title=driver.title)

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Page Text",
            readOnlyHint=True,
            openWorldHint=True,
        )
    )
    async def get_page_text(
        ctx: Context,
        selector: str = "auto",
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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Extracting text from '{selector}'")

        # JavaScript extraction function with smart extraction support
        extraction_script = '''
        function extractFromElement(root) {
            const SKIP_ELEMENTS = new Set([
                'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'SVG', 'CANVAS',
                'IFRAME', 'OBJECT', 'EMBED', 'AUDIO', 'VIDEO', 'MAP', 'HEAD'
            ]);
            const PREFORMATTED_ELEMENTS = new Set(['PRE', 'CODE', 'TEXTAREA']);

            const parts = [];
            let depth = 0;
            let inPreformatted = 0;
            const MAX_DEPTH = 100;

            function walk(node) {
                if (!node) return;
                depth++;
                if (depth > MAX_DEPTH) { depth--; return; }

                try {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const tagName = node.tagName;
                        if (SKIP_ELEMENTS.has(tagName)) return;

                        // Handle IMG elements - emit marker with alt text
                        if (tagName === 'IMG') {
                            const alt = node.getAttribute('alt');
                            const marker = alt && alt.trim()
                                ? '__IMG_ALT__' + alt.trim() + '__END_IMG__'
                                : '__IMG_ALT__(no alt)__END_IMG__';
                            parts.push({text: marker, pre: false});
                            return;  // IMG has no children to walk
                        }

                        const isPre = PREFORMATTED_ELEMENTS.has(tagName) ||
                            (window.getComputedStyle &&
                             ['pre', 'pre-wrap', 'pre-line'].includes(
                                 window.getComputedStyle(node).whiteSpace));

                        if (isPre) inPreformatted++;

                        if (node.shadowRoot) {
                            for (const child of node.shadowRoot.childNodes) {
                                walk(child);
                            }
                        }

                        for (const child of node.childNodes) {
                            walk(child);
                        }

                        if (isPre) inPreformatted--;
                    }
                    else if (node.nodeType === Node.TEXT_NODE) {
                        const text = node.textContent;
                        if (text) {
                            parts.push({text, pre: inPreformatted > 0});
                        }
                    }
                    else if (node.nodeType === Node.DOCUMENT_FRAGMENT_NODE) {
                        for (const child of node.childNodes) {
                            walk(child);
                        }
                    }
                } finally {
                    depth--;
                }
            }

            walk(root);

            let result = '';
            for (const part of parts) {
                if (part.pre) {
                    result += part.text;
                } else {
                    result += part.text.replace(/[\\s\\n\\r\\t\\u00A0]+/g, ' ');
                }
            }
            result = result.trim();

            const MAX_SIZE = 5 * 1024 * 1024;
            if (result.length > MAX_SIZE) {
                result = result.substring(0, MAX_SIZE) + ' [Content truncated at 5MB limit]';
            }

            return result;
        }

        function extractAllText(requestedSelector) {
            const SMART_THRESHOLD = 500;  // Min chars for smart extraction to use an element

            // Calculate body character count for coverage calculation
            const bodyCharCount = extractFromElement(document.body).length;

            // Smart extraction mode
            if (requestedSelector === 'auto') {
                // Priority 1: Try <main> element or [role="main"]
                const main = document.querySelector('main, [role="main"]');
                if (main) {
                    const mainText = extractFromElement(main);
                    if (mainText.length >= SMART_THRESHOLD) {
                        return {
                            text: mainText,
                            title: document.title || '',
                            url: window.location.href,
                            sourceElement: main.tagName.toLowerCase() === 'main' ? 'main' : '[role="main"]',
                            characterCount: mainText.length,
                            isSmartExtraction: true,
                            fallbackUsed: false,
                            bodyCharacterCount: bodyCharCount
                        };
                    }
                }

                // Priority 2: Try <article> element
                const article = document.querySelector('article');
                if (article) {
                    const articleText = extractFromElement(article);
                    if (articleText.length >= SMART_THRESHOLD) {
                        return {
                            text: articleText,
                            title: document.title || '',
                            url: window.location.href,
                            sourceElement: 'article',
                            characterCount: articleText.length,
                            isSmartExtraction: true,
                            fallbackUsed: false,
                            bodyCharacterCount: bodyCharCount
                        };
                    }
                }

                // Fallback: Use body
                const bodyText = extractFromElement(document.body);
                return {
                    text: bodyText,
                    title: document.title || '',
                    url: window.location.href,
                    sourceElement: 'body',
                    characterCount: bodyText.length,
                    isSmartExtraction: true,
                    fallbackUsed: true,
                    bodyCharacterCount: bodyCharCount
                };
            }

            // Explicit selector mode
            const root = document.querySelector(requestedSelector);
            if (!root) {
                return {
                    error: 'Selector not found: ' + requestedSelector,
                    title: document.title || '',
                    url: window.location.href
                };
            }

            const text = extractFromElement(root);
            return {
                text: text,
                title: document.title || '',
                url: window.location.href,
                sourceElement: requestedSelector,
                characterCount: text.length,
                isSmartExtraction: false
            };
        }

        return extractAllText(arguments[0]);
        '''

        # Execute the extraction script
        try:
            result = await asyncio.to_thread(
                driver.execute_script, extraction_script, selector
            )
        except WebDriverException as e:
            raise fastmcp.exceptions.ToolError(
                f"Failed to execute extraction script: {e}"
            )

        # Handle selector not found
        if result.get('error'):
            raise fastmcp.exceptions.ToolError(
                f"Selector '{selector}' not found on page. "
                "Use get_aria_snapshot(selector='body') to discover valid selectors, "
                "or use get_page_html() to inspect the page structure."
            )

        # Extract result components
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        text = result.get('text', '')

        # Handle image markers based on include_images parameter
        if include_images:
            # Convert markers to [Image: alt text] format
            text = re.sub(r'__IMG_ALT__(.+?)__END_IMG__', r'[Image: \1]', text)
        else:
            # Remove all image markers
            text = re.sub(r'__IMG_ALT__.+?__END_IMG__', '', text)

        char_count = len(text)  # Recalculate after marker processing

        # Handle empty extraction
        if not text:
            text = "[No text content found in selected element]"

        # Get actual source element (may differ from selector in auto mode)
        source_element = result.get('sourceElement', selector)

        # Build smart_info only for auto mode
        smart_info = None
        if result.get('isSmartExtraction'):
            smart_info = SmartExtractionInfo(
                fallback_used=result.get('fallbackUsed', False),
                body_character_count=result.get('bodyCharacterCount', 0),
            )

        await logger.info(f"Extracted {char_count:,} characters from <{source_element}>")

        return PageTextResult(
            title=title,
            url=url,
            source_element=source_element,
            text=text,
            character_count=char_count,
            smart_info=smart_info,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Page HTML",
            readOnlyHint=True,
            openWorldHint=True,
        )
    )
    async def get_page_html(
        ctx: Context,
        selector: str | None = None,
        limit: int | None = None,
    ) -> str:
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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        if selector:
            await logger.info(
                f"Extracting HTML for '{selector}'"
                + (f" (limit: {limit})" if limit else "")
            )

            try:
                elements = await asyncio.to_thread(
                    driver.find_elements, By.CSS_SELECTOR, selector
                )
            except WebDriverException as e:
                raise fastmcp.exceptions.ToolError(
                    f"Invalid CSS selector '{selector}': {e}"
                )

            count = len(elements)

            if count == 0:
                raise fastmcp.exceptions.ToolError(
                    f"No elements found matching selector '{selector}'. "
                    "Use get_aria_snapshot(selector='body') to discover valid selectors."
                )

            actual_limit = min(count, limit) if limit is not None else count

            html_parts = []
            for i in range(actual_limit):
                try:
                    html = await asyncio.to_thread(
                        driver.execute_script,
                        "return arguments[0].outerHTML;",
                        elements[i]
                    )
                    html_parts.append(html)
                except WebDriverException:
                    continue  # Element may have become stale

            await logger.info(
                f"Extracted {len(html_parts)} of {count} elements"
                + (f" (limited to {limit})" if limit and count > limit else "")
            )

            return "\n".join(html_parts)

        else:
            await logger.info("Extracting full page HTML source")

            try:
                html = await asyncio.to_thread(lambda: driver.page_source)
            except WebDriverException as e:
                raise fastmcp.exceptions.ToolError(
                    f"Failed to get page source: {e}"
                )

            await logger.info(f"Extracted {len(html):,} characters of HTML")

            return html

    @mcp.tool(annotations=ToolAnnotations(title="Take Screenshot", readOnlyHint=True))
    async def screenshot(filename: str, ctx: Context, full_page: bool = False) -> str:
        """Capture visual screenshot for verification and debugging.

        Use cases:
            - Verify visual appearance after taking actions
            - Debug layout or styling issues
            - Read text rendered in images (not in DOM)
            - Confirm visual state when structure is correct but appearance needs checking

        Args:
            filename: Output filename (e.g., "checkout-page.png")
            ctx: MCP context
            full_page: True for entire scrollable page, False for viewport only

        Returns:
            Absolute path to saved screenshot (use Read tool to view)

        Note: Requires vision processing. full_page=True uses CDP Page.captureScreenshot.
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
        if not url.startswith(("http://", "https://", "file://")):
            raise fastmcp.exceptions.ValidationError(
                "URL must start with http://, https://, or file://"
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
        selector: str, include_urls: bool = False
    ) -> str:
        """Understand page structure and find elements. PRIMARY tool for page comprehension.

        Returns semantic accessibility tree with roles, names, and hierarchy. Use immediately
        after navigate() or any action that changes page content. Provides complete structural
        understanding with 91% token compression vs HTML.

        Args:
            selector: CSS selector scope ('body' for full page, 'form' for specific sections)
            include_urls: Include href values (default False saves tokens)

        Returns:
            YAML with ARIA roles, accessible names, element hierarchy, and states

        Workflow:
            1. Call this after navigate() to understand available elements
            2. Use returned structure to identify elements of interest
            3. Call get_interactive_elements() if you need CSS selectors for clicking
            4. Call get_page_text() if you need actual text content
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

            // Shared whitespace normalization helper
            // Per WAI-ARIA 1.2 and CSS Text Module Level 3
            function normalize(text) {
                return text ? text.replace(/\\s+/g, ' ').trim() : '';
            }

            // Accessible name computation per WAI-ARIA spec
            function computeAccessibleName(el) {
                // Step 1: aria-label
                if (el.getAttribute('aria-label')) {
                    return normalize(el.getAttribute('aria-label'));
                }

                // Step 2: aria-labelledby
                if (el.getAttribute('aria-labelledby')) {
                    const ids = el.getAttribute('aria-labelledby').split(/\\s+/);
                    return ids
                        .map(id => {
                            const refEl = document.getElementById(id);
                            return refEl ? normalize(refEl.textContent) : '';
                        })
                        .filter(Boolean)
                        .join(' ');
                }

                // Step 3: Label element association
                if (el.id) {
                    const label = document.querySelector(`label[for="${el.id}"]`);
                    if (label) return normalize(label.textContent);
                }

                // Step 4: Implicit label (form control inside label)
                if (el.closest('label')) {
                    return normalize(el.closest('label').textContent);
                }

                // Step 5: Element content for links, buttons, headings
                const tagName = el.tagName.toLowerCase();
                if (['button', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
                    return normalize(el.textContent);
                }

                // Step 6: Title attribute
                if (el.getAttribute('title')) {
                    return normalize(el.getAttribute('title'));
                }

                // Step 7: Alt text for images
                if (tagName === 'img') {
                    return normalize(el.getAttribute('alt') || '');
                }

                // Step 8: Placeholder for inputs
                if (['input', 'textarea'].includes(tagName)) {
                    return normalize(el.placeholder || el.value || '');
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
                    const text = normalize(el.textContent);
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
                # Full whitespace normalization: collapse all \s+ to single space
                content = " ".join(node.get("content", "").split())
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
        """Click an element. Auto-waits for visibility and clickability.

        Use selector from get_interactive_elements(). For dynamic content that loads after
        clicking, use wait_for_network_idle() afterward instead of wait_for_network parameter.

        Args:
            selector: CSS selector from get_interactive_elements()
            wait_for_network: Add fixed delay after click (use wait_for_network_idle() instead)
            network_timeout: Delay duration in ms

        Workflow:
            1. get_interactive_elements(text_contains='Submit') - get selector
            2. click(selector) - perform click
            3. wait_for_network_idle() - wait for dynamic content
            4. get_aria_snapshot() - understand new page state
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

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Capture Core Web Vitals",
            readOnlyHint=True,
            idempotentHint=True,
        )
    )
    async def capture_web_vitals(
        ctx: Context,
        timeout_ms: int = 5000,
    ) -> CoreWebVitals:
        """Capture Core Web Vitals (LCP, CLS, INP, FCP, TTFB) for current page.

        Uses JavaScript Performance APIs to collect metrics post-navigation.
        Must be called after navigate().

        Args:
            ctx: MCP context
            timeout_ms: Max wait time for metrics in milliseconds (default 5000)

        Returns:
            CoreWebVitals with all available metrics and ratings

        Core Web Vitals (2025):
        - LCP (Largest Contentful Paint): ≤2.5s good, >4.0s poor
        - INP (Interaction to Next Paint): ≤200ms good, >500ms poor
        - CLS (Cumulative Layout Shift): <0.1 good, >0.25 poor

        Supplementary:
        - FCP (First Contentful Paint): ≤1.8s good, >3.0s poor
        - TTFB (Time to First Byte): ≤0.8s good, >1.8s poor

        Note: INP requires user interaction to measure. LCP may not be final
              until user interacts with the page.
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        start_time = time.time()
        current_url = driver.current_url

        await logger.info(f"Capturing Core Web Vitals for {current_url}")

        # JavaScript to collect all Web Vitals using Performance APIs
        # Uses buffered: true to capture metrics that occurred before script runs
        script = """
        var callback = arguments[arguments.length - 1];
        var timeoutMs = arguments[0];

        (async function collectWebVitals() {
            var results = { fcp: null, lcp: null, ttfb: null, cls: null, inp: null };

            // FCP - immediate from paint entries
            try {
                var fcpEntry = performance.getEntriesByName('first-contentful-paint', 'paint')[0];
                if (fcpEntry) {
                    results.fcp = {
                        name: 'FCP',
                        value: fcpEntry.startTime,
                        rating: fcpEntry.startTime <= 1800 ? 'good' : fcpEntry.startTime <= 3000 ? 'needs-improvement' : 'poor'
                    };
                }
            } catch (e) { results.fcp = { error: e.toString() }; }

            // TTFB - immediate from navigation timing
            try {
                var navEntries = performance.getEntriesByType('navigation');
                var navEntry = navEntries[0];
                if (navEntry) {
                    results.ttfb = {
                        name: 'TTFB',
                        value: navEntry.responseStart,
                        rating: navEntry.responseStart <= 800 ? 'good' : navEntry.responseStart <= 1800 ? 'needs-improvement' : 'poor',
                        phases: {
                            dns: navEntry.domainLookupEnd - navEntry.domainLookupStart,
                            tcp: navEntry.connectEnd - navEntry.connectStart,
                            request: navEntry.responseStart - navEntry.requestStart
                        }
                    };
                }
            } catch (e) { results.ttfb = { error: e.toString() }; }

            // LCP - use PerformanceObserver with buffered flag
            try {
                results.lcp = await new Promise(function(resolve) {
                    var lastEntry = null;
                    var observer = new PerformanceObserver(function(list) {
                        var entries = list.getEntries();
                        lastEntry = entries[entries.length - 1];
                    });
                    observer.observe({ type: 'largest-contentful-paint', buffered: true });
                    setTimeout(function() {
                        observer.disconnect();
                        if (lastEntry) {
                            resolve({
                                name: 'LCP',
                                value: lastEntry.startTime,
                                size: lastEntry.size,
                                element_id: lastEntry.id || null,
                                url: lastEntry.url || null,
                                rating: lastEntry.startTime <= 2500 ? 'good' : lastEntry.startTime <= 4000 ? 'needs-improvement' : 'poor'
                            });
                        } else { resolve(null); }
                    }, Math.min(timeoutMs, 3000));
                });
            } catch (e) { results.lcp = { error: e.toString() }; }

            // CLS - collect layout shifts
            try {
                results.cls = await new Promise(function(resolve) {
                    var sessionValue = 0;
                    var sessionEntries = [];
                    var observer = new PerformanceObserver(function(list) {
                        var entries = list.getEntries();
                        for (var i = 0; i < entries.length; i++) {
                            var entry = entries[i];
                            if (!entry.hadRecentInput) {
                                sessionValue += entry.value;
                                var sources = [];
                                if (entry.sources) {
                                    for (var j = 0; j < entry.sources.length; j++) {
                                        var s = entry.sources[j];
                                        sources.push({
                                            node: s.node ? s.node.tagName : null
                                        });
                                    }
                                }
                                sessionEntries.push({
                                    value: entry.value,
                                    time: entry.startTime,
                                    sources: sources
                                });
                            }
                        }
                    });
                    observer.observe({ type: 'layout-shift', buffered: true });
                    setTimeout(function() {
                        observer.disconnect();
                        resolve({
                            name: 'CLS',
                            value: sessionValue,
                            rating: sessionValue <= 0.1 ? 'good' : sessionValue <= 0.25 ? 'needs-improvement' : 'poor',
                            entries: sessionEntries
                        });
                    }, Math.min(timeoutMs, 2000));
                });
            } catch (e) { results.cls = { error: e.toString() }; }

            // INP - collect event timing (requires user interaction)
            try {
                results.inp = await new Promise(function(resolve) {
                    var worstInteraction = null;
                    var observer = new PerformanceObserver(function(list) {
                        var entries = list.getEntries();
                        for (var i = 0; i < entries.length; i++) {
                            var entry = entries[i];
                            if (!worstInteraction || entry.duration > worstInteraction.duration) {
                                worstInteraction = {
                                    duration: entry.duration,
                                    name: entry.name,
                                    start_time: entry.startTime,
                                    input_delay: entry.processingStart - entry.startTime,
                                    processing_time: entry.processingEnd - entry.processingStart,
                                    presentation_delay: entry.duration - (entry.processingEnd - entry.startTime)
                                };
                            }
                        }
                    });
                    observer.observe({ type: 'event', durationThreshold: 40, buffered: true });
                    setTimeout(function() {
                        observer.disconnect();
                        if (worstInteraction) {
                            resolve({
                                name: 'INP',
                                value: worstInteraction.duration,
                                rating: worstInteraction.duration <= 200 ? 'good' : worstInteraction.duration <= 500 ? 'needs-improvement' : 'poor',
                                details: worstInteraction
                            });
                        } else { resolve(null); }
                    }, Math.min(timeoutMs, 1000));
                });
            } catch (e) { results.inp = { error: e.toString() }; }

            return results;
        })().then(callback).catch(function(err) { callback({ error: err.toString() }); });
        """

        errors = []
        try:
            # Use execute_async_script for Promise-based collection
            results = await asyncio.to_thread(
                driver.execute_async_script,
                script,
                timeout_ms,
            )
        except Exception as e:
            await logger.info(f"Web Vitals collection failed: {e}")
            errors.append(f"Script execution failed: {e}")
            results = {}

        collection_duration = (time.time() - start_time) * 1000

        # Parse results into Pydantic models
        def parse_metric(data, model_cls):
            if not data:
                return None
            if isinstance(data, dict) and "error" in data:
                errors.append(f"{data.get('name', 'Unknown')}: {data['error']}")
                return None
            try:
                return model_cls(**data)
            except Exception as e:
                errors.append(f"Failed to parse {model_cls.__name__}: {e}")
                return None

        vitals = CoreWebVitals(
            url=current_url,
            timestamp=time.time(),
            fcp=parse_metric(results.get("fcp") if results else None, FCPMetric),
            lcp=parse_metric(results.get("lcp") if results else None, LCPMetric),
            ttfb=parse_metric(results.get("ttfb") if results else None, TTFBMetric),
            cls=parse_metric(results.get("cls") if results else None, CLSMetric),
            inp=parse_metric(results.get("inp") if results else None, INPMetric),
            collection_duration_ms=collection_duration,
            errors=errors,
        )

        # Log summary
        metrics_found = []
        if vitals.fcp:
            metrics_found.append(f"FCP={vitals.fcp.value:.0f}ms ({vitals.fcp.rating})")
        if vitals.lcp:
            metrics_found.append(f"LCP={vitals.lcp.value:.0f}ms ({vitals.lcp.rating})")
        if vitals.ttfb:
            metrics_found.append(f"TTFB={vitals.ttfb.value:.0f}ms ({vitals.ttfb.rating})")
        if vitals.cls:
            metrics_found.append(f"CLS={vitals.cls.value:.3f} ({vitals.cls.rating})")
        if vitals.inp:
            metrics_found.append(f"INP={vitals.inp.value:.0f}ms ({vitals.inp.rating})")

        await logger.info(f"Web Vitals captured in {collection_duration:.0f}ms: {', '.join(metrics_found) or 'none'}")

        return vitals


    @mcp.tool(
        description="""Enable network timing capture via CDP.

Call this before navigate() or click() to capture network requests.
Then call get_network_timings() to retrieve the captured data.

Example workflow:
1. start_network_capture()
2. navigate(url)
3. click(selector)  # optional - more requests captured
4. get_network_timings()  # returns all requests with timing

Args:
    resource_types: Optional filter (e.g., ["fetch", "xmlhttprequest"]).
                   If None, captures all resource types.

Returns:
    Status dict with enabled state and capture start timestamp."""
    )
    async def start_network_capture(
        ctx: Context,
        resource_types: list[str] | None = None,
    ) -> dict:
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        if service.state.network_capture_enabled:
            await logger.info("Network capture already enabled")
            return {
                "enabled": True,
                "capturing_since": service.state.network_capture_start_time,
                "note": "Already capturing",
            }

        # Enable CDP Network domain with large buffers for HAR export
        # Per Perplexity research: Chrome can handle 50MB/resource, 100MB total
        await asyncio.to_thread(
            driver.execute_cdp_cmd,
            "Network.enable",
            {
                "maxTotalBufferSize": 100 * 1024 * 1024,  # 100MB total
                "maxResourceBufferSize": 50 * 1024 * 1024,  # 50MB per resource
            },
        )

        # Clear any existing performance entries for clean capture
        await asyncio.to_thread(
            driver.execute_script,
            "performance.clearResourceTimings();",
        )

        # Update state
        service.state.network_capture_enabled = True
        service.state.network_capture_start_time = time.time()
        service.state.network_events = []
        service.state.network_resource_filter = (
            [t.lower() for t in resource_types] if resource_types else None
        )

        await logger.info(
            f"Network capture enabled"
            + (f" (filtering: {resource_types})" if resource_types else "")
        )

        return {
            "enabled": True,
            "capturing_since": service.state.network_capture_start_time,
            "resource_filter": resource_types,
        }

    @mcp.tool(
        description="""Retrieve captured network timing data.

Returns all network requests since start_network_capture() was called,
with detailed timing breakdown for each request. Useful for identifying
slow API calls and network bottlenecks.

Args:
    clear: If True (default), clears captured data after retrieval.
    min_duration_ms: Only include requests slower than this threshold (0 = all).

Returns:
    NetworkCapture with requests, timing breakdown, and summary statistics
    including slowest requests and breakdown by resource type."""
    )
    async def get_network_timings(
        ctx: Context,
        clear: bool = True,
        min_duration_ms: int = 0,
    ) -> NetworkCapture:
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        start_time = time.time()
        current_url = driver.current_url

        if not service.state.network_capture_enabled:
            await logger.info("Network capture not enabled - returning empty result")
            return NetworkCapture(
                url=current_url,
                timestamp=time.time(),
                requests=[],
                total_requests=0,
                total_size_bytes=0,
                total_time_ms=0,
                errors=["Network capture not enabled. Call start_network_capture() first."],
            )

        # Collect resource timing via JavaScript Performance API
        # This captures all network requests with detailed timing breakdown
        script = """
        var entries = performance.getEntriesByType('resource');
        return entries.map(function(r) {
            return {
                url: r.name,
                method: 'GET',
                resource_type: r.initiatorType,
                started_at: r.startTime,
                duration_ms: r.duration,
                encoded_data_length: r.transferSize || 0,
                timing: {
                    blocked: Math.max(0, r.fetchStart - r.startTime),
                    dns: Math.max(0, r.domainLookupEnd - r.domainLookupStart),
                    connect: Math.max(0, r.connectEnd - r.connectStart),
                    ssl: r.secureConnectionStart > 0 ? Math.max(0, r.connectEnd - r.secureConnectionStart) : 0,
                    send: Math.max(0, r.requestStart - r.connectEnd),
                    wait: Math.max(0, r.responseStart - r.requestStart),
                    receive: Math.max(0, r.responseEnd - r.responseStart)
                }
            };
        });
        """

        raw_entries = await asyncio.to_thread(driver.execute_script, script)

        # Apply resource type filter if set
        resource_filter = service.state.network_resource_filter
        if resource_filter:
            raw_entries = [
                e for e in raw_entries
                if e.get("resource_type", "").lower() in resource_filter
            ]

        # Filter by min_duration_ms
        if min_duration_ms > 0:
            raw_entries = [
                e for e in raw_entries if e.get("duration_ms", 0) >= min_duration_ms
            ]

        # Convert to NetworkRequest models
        requests = []
        total_size = 0
        for i, entry in enumerate(raw_entries):
            timing_data = entry.get("timing", {})
            req = NetworkRequest(
                request_id=str(i),
                url=entry.get("url", ""),
                method=entry.get("method", "GET"),
                resource_type=entry.get("resource_type"),
                encoded_data_length=entry.get("encoded_data_length", 0),
                started_at=entry.get("started_at", 0),
                duration_ms=entry.get("duration_ms"),
                timing=RequestTiming(
                    blocked=timing_data.get("blocked", 0),
                    dns=timing_data.get("dns", 0),
                    connect=timing_data.get("connect", 0),
                    ssl=timing_data.get("ssl", 0),
                    send=timing_data.get("send", 0),
                    wait=timing_data.get("wait", 0),
                    receive=timing_data.get("receive", 0),
                )
                if timing_data
                else None,
            )
            requests.append(req)
            total_size += entry.get("encoded_data_length", 0)

        # Sort by duration (slowest first) for summary
        sorted_by_duration = sorted(
            requests, key=lambda r: r.duration_ms or 0, reverse=True
        )
        slowest = [
            {"url": r.url, "duration_ms": r.duration_ms, "type": r.resource_type}
            for r in sorted_by_duration[:10]
        ]

        # Count by resource type
        type_counts: dict[str, int] = {}
        for r in requests:
            t = r.resource_type or "other"
            type_counts[t] = type_counts.get(t, 0) + 1

        # Calculate total time (longest request duration)
        total_time = max((r.duration_ms or 0) for r in requests) if requests else 0

        # Clear resource timing buffer if requested
        if clear:
            await asyncio.to_thread(
                driver.execute_script,
                "performance.clearResourceTimings();",
            )

        collection_duration = (time.time() - start_time) * 1000

        result = NetworkCapture(
            url=current_url,
            timestamp=time.time(),
            requests=requests,
            total_requests=len(requests),
            total_size_bytes=total_size,
            total_time_ms=total_time,
            slowest_requests=slowest,
            requests_by_type=type_counts,
            errors=[],
        )

        # Log summary
        if slowest:
            slowest_url = slowest[0]["url"]
            # Truncate URL for logging
            if len(slowest_url) > 60:
                slowest_url = slowest_url[:57] + "..."
            await logger.info(
                f"Captured {len(requests)} requests in {collection_duration:.0f}ms, "
                f"slowest: {slowest[0]['duration_ms']:.0f}ms ({slowest_url})"
            )
        else:
            await logger.info(
                f"Captured {len(requests)} requests in {collection_duration:.0f}ms"
            )

        return result

    @mcp.tool(
        description="""Export captured network traffic to HAR 1.2 file.

Exports full HTTP transaction details (headers, status codes, timing) to HAR format
for analysis in Chrome DevTools or other tools.

IMPORTANT: HAR capture must be enabled via navigate(enable_har_capture=True, fresh_browser=True).
This is opt-in due to performance overhead from Chrome's performance logging.

Args:
    filename: Output filename (required, e.g., "api-calls.har")
    include_response_bodies: If True, fetch response bodies for JSON/text (default False)
    max_body_size_mb: Max response body size to fetch in MB (default 10, max 50)

Returns:
    HARExportResult with path to saved file and entry count

Workflow:
    1. navigate(url, fresh_browser=True, enable_har_capture=True) - enable HAR capture
    2. [interact with page]
    3. export_har("capture.har") - export network data
    4. Read the HAR file or import into Chrome DevTools"""
    )
    async def export_har(
        ctx: Context,
        filename: str,
        include_response_bodies: bool = False,
        max_body_size_mb: int = 10,
    ) -> HARExportResult:
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Exporting HAR to {filename}")
        errors: list[str] = []

        # Get performance logs (clears buffer - subsequent calls return only newer entries)
        try:
            logs = await asyncio.to_thread(driver.get_log, "performance")
        except WebDriverException as e:
            errors.append(f"Failed to get performance logs: {e}")
            logs = []

        if not logs:
            await logger.info("No performance logs available")
            har_path = service.state.capture_dir / filename
            empty_har = {
                "log": {
                    "version": "1.2",
                    "creator": {"name": "selenium-browser-automation", "version": "1.0"},
                    "entries": [],
                }
            }
            har_path.write_text(json.dumps(empty_har, indent=2))
            return HARExportResult(
                path=str(har_path),
                entry_count=0,
                size_bytes=len(json.dumps(empty_har)),
                has_errors=True,
                errors=["No performance logs available. Navigate to pages first."],
            )

        # Parse and filter Network.* events
        transactions: dict[str, dict] = {}
        for entry in logs:
            try:
                log_data = json.loads(entry["message"])
                message = log_data.get("message", {})
                method = message.get("method", "")
                params = message.get("params", {})

                if not method.startswith("Network."):
                    continue

                request_id = params.get("requestId")
                if not request_id:
                    continue

                if method == "Network.requestWillBeSent":
                    req = params.get("request", {})
                    transactions[request_id] = {
                        "request": req,
                        "wall_time": params.get("wallTime"),
                        "timestamp": params.get("timestamp"),
                        "initiator": params.get("initiator"),
                    }

                elif method == "Network.responseReceived":
                    if request_id in transactions:
                        resp = params.get("response", {})
                        transactions[request_id]["response"] = resp
                        transactions[request_id]["resource_type"] = params.get("type")

                elif method == "Network.loadingFinished":
                    if request_id in transactions:
                        transactions[request_id]["encoded_data_length"] = params.get(
                            "encodedDataLength", 0
                        )
                        transactions[request_id]["complete"] = True

            except (json.JSONDecodeError, KeyError) as e:
                continue

        # Convert transactions to HAR entries
        har_entries = []

        for request_id, txn in transactions.items():
            if "response" not in txn:
                continue  # Skip incomplete transactions

            req = txn.get("request", {})
            resp = txn.get("response", {})
            timing = resp.get("timing", {})

            # Convert wallTime (seconds since epoch) to ISO 8601
            wall_time = txn.get("wall_time")
            if wall_time:
                dt = datetime.fromtimestamp(wall_time, tz=timezone.utc)
                started_datetime = dt.isoformat().replace("+00:00", "Z")
            else:
                started_datetime = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            # Convert headers dict to HAR array format
            def headers_to_har(headers_dict):
                if not headers_dict:
                    return []
                return [{"name": k, "value": str(v)} for k, v in headers_dict.items()]

            # Parse query string from URL
            def parse_query_string(url):
                parsed = urlparse(url)
                query_params = parse_qs(parsed.query)
                result = []
                for name, values in query_params.items():
                    for value in values:
                        result.append({"name": name, "value": value})
                return result

            # Convert CDP timing to HAR timing (duration in ms)
            def convert_timing(timing_obj):
                def safe_duration(start_key, end_key):
                    start = timing_obj.get(start_key)
                    end = timing_obj.get(end_key)
                    if start is not None and end is not None and start >= 0 and end >= 0:
                        return max(0, end - start)
                    return -1

                return {
                    "blocked": -1,  # Not directly available
                    "dns": safe_duration("dnsStart", "dnsEnd"),
                    "connect": safe_duration("connectStart", "connectEnd"),
                    "ssl": safe_duration("sslStart", "sslEnd"),
                    "send": safe_duration("sendStart", "sendEnd"),
                    "wait": safe_duration("sendEnd", "receiveHeadersEnd"),
                    "receive": 0,  # Would need loadingFinished timing
                }

            har_timing = convert_timing(timing)

            # Calculate total time
            total_time = sum(v for v in har_timing.values() if v >= 0)

            har_entry = {
                "startedDateTime": started_datetime,
                "time": total_time,
                "request": {
                    "method": req.get("method", "GET"),
                    "url": req.get("url", ""),
                    "httpVersion": resp.get("protocol", "HTTP/1.1"),
                    "headers": headers_to_har(req.get("headers", {})),
                    "queryString": parse_query_string(req.get("url", "")),
                    "cookies": [],  # Would need to parse from headers
                    "headersSize": -1,
                    "bodySize": len(req.get("postData", "")) if req.get("postData") else 0,
                },
                "response": {
                    "status": resp.get("status", 0),
                    "statusText": resp.get("statusText", ""),
                    "httpVersion": resp.get("protocol", "HTTP/1.1"),
                    "headers": headers_to_har(resp.get("headers", {})),
                    "cookies": [],
                    "content": {
                        "size": resp.get("encodedDataLength", 0),
                        "mimeType": resp.get("mimeType", ""),
                    },
                    "redirectURL": "",
                    "headersSize": -1,
                    "bodySize": txn.get("encoded_data_length", -1),
                },
                "cache": {},
                "timings": har_timing,
                "serverIPAddress": resp.get("remoteIPAddress", ""),
                "connection": resp.get("connectionId", ""),
            }

            # Add POST data if present
            if req.get("postData"):
                har_entry["request"]["postData"] = {
                    "mimeType": req.get("headers", {}).get("Content-Type", ""),
                    "text": req.get("postData"),
                }

            # Fetch response body if requested (configurable size limit)
            if include_response_bodies:
                mime_type = resp.get("mimeType", "")
                body_size = txn.get("encoded_data_length", 0)
                max_body_bytes = max(1, min(max_body_size_mb, 50)) * 1024 * 1024  # 1-50MB
                should_fetch = (
                    body_size < max_body_bytes
                    and (
                        "json" in mime_type
                        or "text" in mime_type
                        or "xml" in mime_type
                        or "javascript" in mime_type
                    )
                )
                if should_fetch:
                    try:
                        body_result = await asyncio.to_thread(
                            driver.execute_cdp_cmd,
                            "Network.getResponseBody",
                            {"requestId": request_id},
                        )
                        if body_result.get("body"):
                            har_entry["response"]["content"]["text"] = body_result["body"]
                            if body_result.get("base64Encoded"):
                                har_entry["response"]["content"]["encoding"] = "base64"
                    except Exception:
                        pass  # Body may not be available after the fact

            har_entries.append(har_entry)

        # Build HAR structure
        har = {
            "log": {
                "version": "1.2",
                "creator": {"name": "selenium-browser-automation", "version": "1.0"},
                "entries": har_entries,
            }
        }

        # Save to file
        har_path = service.state.capture_dir / filename
        har_json = json.dumps(har, indent=2)
        har_path.write_text(har_json)

        await logger.info(f"Exported {len(har_entries)} entries to {har_path}")

        return HARExportResult(
            path=str(har_path),
            entry_count=len(har_entries),
            size_bytes=len(har_json),
            has_errors=len(errors) > 0,
            errors=errors,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Configure Proxy",
            destructiveHint=False,
            idempotentHint=False,
        )
    )
    async def configure_proxy(
        host: str,
        port: int,
        username: str,
        password: str,
        ctx: Context | None = None,
    ) -> dict:
        """Configure authenticated HTTP proxy for bypassing IP-based rate limiting.

        Starts a local mitmproxy instance that handles authentication with the
        upstream proxy. Chrome connects to the local mitmproxy (no auth needed),
        and mitmproxy forwards requests with credentials to Bright Data.

        Architecture:
            Chrome → localhost:8080 (mitmproxy) → upstream proxy (Bright Data)
                     [no auth needed]              [mitmproxy handles auth]

        IP Rotation Behavior:
            - Same browser session = same IP (connection reuse)
            - navigate(url, fresh_browser=True) = NEW IP from proxy pool
            - Calling configure_proxy() again = new IP (restarts mitmproxy)

        Rate Limit Bypass Pattern:
            For sites with aggressive rate limiting, use fresh_browser=True
            on each navigate() call to get a new IP for each request.

        Args:
            host: Proxy host (e.g., brd.superproxy.io)
            port: Proxy port (e.g., 33335 for HTTP, 22228 for SOCKS5)
            username: Proxy username (Bright Data format: brd-customer-{ID}-zone-{ZONE}-country-{CC})
            password: Proxy password

        Returns:
            Status dict confirming proxy configuration

        Requires:
            mitmproxy must be installed: brew install mitmproxy (macOS) or pip install mitmproxy

        Example:
            configure_proxy(
                host="brd.superproxy.io",
                port=33335,
                username="brd-customer-hl_87cafc8d-zone-residential-country-us",
                password="YOUR_PASSWORD"
            )
            navigate("https://api.ipify.org")  # Shows proxy IP, not your real IP
            navigate(url, fresh_browser=True)  # Gets NEW IP from proxy pool
        """
        if ctx:
            logger = PrintLogger(ctx)
            await logger.info(f"Configuring proxy via mitmproxy: {host}:{port}")

        # Close existing browser and mitmproxy
        await service.close_browser()

        # Stop any existing mitmproxy process
        if service.state.mitmproxy_process:
            service.state.mitmproxy_process.terminate()
            try:
                service.state.mitmproxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service.state.mitmproxy_process.kill()
            service.state.mitmproxy_process = None

        # Store proxy config
        service.state.proxy_config = {
            'host': host,
            'port': str(port),
            'username': username,
            'password': password,
        }

        # Start mitmproxy with upstream authentication
        # mitmproxy handles auth with Bright Data, Chrome connects to localhost:8080
        upstream_url = f"http://{host}:{port}"
        upstream_auth = f"{username}:{password}"

        try:
            service.state.mitmproxy_process = subprocess.Popen(
                [
                    'mitmdump',
                    '--mode', f'upstream:{upstream_url}',
                    '--upstream-auth', upstream_auth,
                    '--listen-host', '127.0.0.1',
                    '--listen-port', '8080',
                    '--ssl-insecure',  # Accept upstream proxy's certs
                    '--quiet',  # Reduce log noise
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Give mitmproxy time to start
            await asyncio.sleep(2)

            # Check if process is still running
            if service.state.mitmproxy_process.poll() is not None:
                stderr = service.state.mitmproxy_process.stderr.read().decode() if service.state.mitmproxy_process.stderr else ""
                raise RuntimeError(f"mitmproxy failed to start: {stderr}")

            if ctx:
                await logger.info("mitmproxy started on localhost:8080")

        except FileNotFoundError:
            service.state.proxy_config = None
            raise fastmcp.exceptions.ToolError(
                "mitmproxy not found. Install with: pip install mitmproxy"
            )
        except Exception as e:
            service.state.proxy_config = None
            if service.state.mitmproxy_process:
                service.state.mitmproxy_process.kill()
                service.state.mitmproxy_process = None
            raise fastmcp.exceptions.ToolError(f"Failed to start mitmproxy: {e}")

        return {
            "status": "proxy_configured",
            "host": host,
            "port": port,
            "note": "Browser will use this proxy on next navigate()"
        }

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Clear Proxy",
            destructiveHint=False,
            idempotentHint=True,
        )
    )
    async def clear_proxy(ctx: Context | None = None) -> dict:
        """Clear proxy configuration and return to direct connection.

        Stops the mitmproxy subprocess and clears proxy settings.

        Returns:
            Status dict confirming proxy cleared
        """
        if ctx:
            logger = PrintLogger(ctx)
            await logger.info("Clearing proxy configuration")

        # Close browser first
        await service.close_browser()

        # Stop mitmproxy process
        if service.state.mitmproxy_process:
            service.state.mitmproxy_process.terminate()
            try:
                service.state.mitmproxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                service.state.mitmproxy_process.kill()
            service.state.mitmproxy_process = None
            if ctx:
                await logger.info("mitmproxy stopped")

        service.state.proxy_config = None

        return {"status": "proxy_cleared"}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Execute JavaScript",
            readOnlyHint=False,
            idempotentHint=False,
        )
    )
    async def execute_javascript(
        code: str,
        ctx: Context,
        timeout_ms: int = 5000,
    ) -> JavaScriptResult:
        """Execute JavaScript in the browser and return the result.

        Evaluates a JavaScript expression in the current page context.
        For multiple statements, wrap in an IIFE (Immediately Invoked Function Expression).

        Args:
            code: JavaScript expression to evaluate.
                  For multiple statements: (() => { const x = 1; return x; })()
            timeout_ms: Maximum execution time in milliseconds. 0 disables timeout. Default: 5000.

        Returns:
            JavaScriptResult with success status, typed result, and error details if failed.

        Examples:
            # Simple expression
            execute_javascript(code="document.title")

            # Object literal (parentheses required)
            execute_javascript(code="({url: location.href, links: document.links.length})")

            # Next.js data extraction
            execute_javascript(code="JSON.parse(document.getElementById('__NEXT_DATA__')?.textContent || '{}')")

            # Async/Promise (automatically awaited)
            execute_javascript(code="fetch('/api').then(r => r.json())")

            # Multiple statements with IIFE
            execute_javascript(code="(() => { const el = document.querySelector('h1'); return el?.textContent; })()")

            # Install fetch interceptor
            execute_javascript(code='''(() => {
                window.__fetchCapture = [];
                const orig = window.fetch;
                window.fetch = async (...args) => {
                    window.__fetchCapture.push({url: args[0], time: Date.now()});
                    return orig.apply(window, args);
                };
                return "interceptor installed";
            })()''')

        Notes:
            - Promises are automatically awaited
            - DOM nodes and functions cannot be serialized (return null with explanation)
            - BigInt and Symbol are converted to strings
            - Map → object, Set → array, circular references → "[Circular Reference]"
            - Sites with strict CSP may block execution
        """
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Executing JavaScript ({len(code)} chars)")

        # JavaScript wrapper with comprehensive serialization handling
        # Key design decisions:
        # 1. safeSerialize handles all JS types that JSON.stringify can't
        # 2. Thenable detection uses typeof check (more robust than instanceof for cross-context)
        # 3. WeakSet tracks seen objects to detect circular references
        # 4. User code is wrapped in new Function to provide arguments access
        wrapper_script = '''
(async function(__userCode) {
    // Helper to safely serialize any JavaScript value to JSON-compatible format
    function safeSerialize(value) {
        // Handle primitives and special types before JSON.stringify
        if (value === undefined) {
            return { success: true, result: null, result_type: 'undefined' };
        }
        if (value === null) {
            return { success: true, result: null, result_type: 'null' };
        }
        if (typeof value === 'function') {
            return { success: true, result: null, result_type: 'function', note: 'Functions cannot be serialized' };
        }
        if (typeof value === 'symbol') {
            return { success: true, result: value.toString(), result_type: 'symbol' };
        }
        if (typeof value === 'bigint') {
            return { success: true, result: value.toString(), result_type: 'bigint' };
        }
        // Handle special number values that JSON.stringify converts to null
        // Return string representations for AI visibility (matches CiC behavior)
        if (typeof value === 'number') {
            if (Number.isNaN(value)) {
                return { success: true, result: 'NaN', result_type: 'number', note: 'Value is NaN (not serializable to JSON)' };
            }
            if (!Number.isFinite(value)) {
                const repr = value > 0 ? 'Infinity' : '-Infinity';
                return { success: true, result: repr, result_type: 'number', note: `Value is ${repr} (not serializable to JSON)` };
            }
            if (Object.is(value, -0)) {
                return { success: true, result: '-0', result_type: 'number', note: 'Value is negative zero (-0)' };
            }
        }
        // DOM nodes cannot be serialized
        if (typeof Node !== 'undefined' && value instanceof Node) {
            return { success: true, result: null, result_type: 'unserializable', note: 'DOM nodes cannot be serialized' };
        }
        // Window object check (common mistake)
        if (typeof Window !== 'undefined' && value instanceof Window) {
            return { success: true, result: null, result_type: 'unserializable', note: 'Window object cannot be serialized' };
        }
        // RegExp converts to string representation
        if (value instanceof RegExp) {
            return { success: true, result: value.toString(), result_type: 'string' };
        }
        // Date converts to ISO string
        if (value instanceof Date) {
            return { success: true, result: value.toISOString(), result_type: 'string' };
        }
        // Map converts to object
        if (value instanceof Map) {
            return safeSerialize(Object.fromEntries(value));
        }
        // Set converts to array
        if (value instanceof Set) {
            return safeSerialize([...value]);
        }
        // WeakMap and WeakSet cannot be serialized - entries are non-enumerable by design
        // This is an ECMAScript specification constraint to prevent observable non-determinism from GC
        if (value instanceof WeakMap) {
            return { success: true, result: null, result_type: 'unserializable', note: 'WeakMap entries cannot be enumerated or serialized. WeakMap is designed for internal object metadata and deliberately prevents access to its contents.' };
        }
        if (value instanceof WeakSet) {
            return { success: true, result: null, result_type: 'unserializable', note: 'WeakSet entries cannot be enumerated or serialized. WeakSet is designed for tracking object membership and deliberately prevents access to its contents.' };
        }
        // Error objects - extract message, name, stack (would otherwise serialize to {})
        if (value instanceof Error) {
            return {
                success: true,
                result: { name: value.name, message: value.message, stack: value.stack || null },
                result_type: 'error'
            };
        }
        // ArrayBuffer cannot be directly serialized
        if (value instanceof ArrayBuffer) {
            return { success: true, result: null, result_type: 'unserializable', note: 'ArrayBuffer cannot be directly serialized. Use TypedArray (e.g., new Uint8Array(buffer)) to access contents.' };
        }
        // Blob contents require async reading
        if (typeof Blob !== 'undefined' && value instanceof Blob) {
            return { success: true, result: null, result_type: 'unserializable', note: 'Blob contents require async reading via blob.text() or blob.arrayBuffer().' };
        }
        // Generator objects cannot be serialized (have next() and Symbol.iterator)
        if (value && typeof value.next === 'function' && typeof value[Symbol.iterator] === 'function') {
            return { success: true, result: null, result_type: 'unserializable', note: 'Generator state cannot be serialized. Consume the generator and return the values instead.' };
        }

        // For objects and arrays, use JSON.stringify with circular reference detection
        const seen = new WeakSet();
        try {
            const serialized = JSON.stringify(value, function(key, val) {
                // Handle BigInt in nested objects
                if (typeof val === 'bigint') {
                    return val.toString();
                }
                // Handle special number values that JSON.stringify converts to null
                if (typeof val === 'number') {
                    if (Number.isNaN(val)) return '[NaN]';
                    if (!Number.isFinite(val)) return val > 0 ? '[Infinity]' : '[-Infinity]';
                    if (Object.is(val, -0)) return '[-0]';
                }
                // Detect circular references
                if (typeof val === 'object' && val !== null) {
                    if (seen.has(val)) {
                        return '[Circular Reference]';
                    }
                    seen.add(val);
                }
                return val;
            });

            // Parse back to get clean object (removes undefined values, etc.)
            const result = JSON.parse(serialized);
            const resultType = Array.isArray(value) ? 'array' : typeof value;
            return { success: true, result: result, result_type: resultType };
        } catch (e) {
            // JSON.stringify failed (shouldn't happen after our checks, but be safe)
            return { success: true, result: null, result_type: 'unserializable', note: e.message };
        }
    }

    try {
        // Try expression form first (most common case)
        // This handles: 1 + 1, document.title, Promise.resolve(42), () => value
        let result;

        try {
            const exprFn = new Function('return (' + __userCode + ')');
            result = exprFn();
        } catch (parseErr) {
            if (parseErr instanceof SyntaxError) {
                // Expression parsing failed, try as statement block
                // This handles: throw new Error(), if/for/while, var x = 1
                const stmtFn = new Function(__userCode);
                result = stmtFn();  // undefined for statements without return
            } else {
                throw parseErr;  // Re-throw non-syntax errors
            }
        }

        // Auto-await if result is thenable (Promise or Promise-like)
        // Use typeof check instead of instanceof for cross-context compatibility
        if (result && typeof result.then === 'function') {
            result = await result;
        }

        return safeSerialize(result);
    } catch (e) {
        return {
            success: false,
            result: null,
            result_type: 'unserializable',
            error: e.message || String(e),
            error_stack: e.stack || null,
            error_type: 'execution'
        };
    }
})
'''

        # Escape user code as JSON string to prevent injection
        # json.dumps handles quotes, backslashes, newlines, etc.
        escaped_code = json.dumps(code)

        # Build the async wrapper for Selenium's execute_async_script
        # The callback is always the last argument provided by Selenium
        async_script = f'''
const callback = arguments[arguments.length - 1];
const userCode = {escaped_code};

{wrapper_script}(userCode)
    .then(function(result) {{
        callback(result);
    }})
    .catch(function(e) {{
        callback({{
            success: false,
            error: e.message || String(e),
            error_type: 'execution'
        }});
    }});
'''

        try:
            # Use asyncio.wait_for for Python-side timeout
            # timeout_ms=0 means no timeout (wait indefinitely)
            if timeout_ms > 0:
                result = await asyncio.wait_for(
                    asyncio.to_thread(driver.execute_async_script, async_script),
                    timeout=timeout_ms / 1000
                )
            else:
                result = await asyncio.to_thread(
                    driver.execute_async_script, async_script
                )

            # Log result type for debugging
            result_type = result.get('result_type', 'unknown') if isinstance(result, dict) else 'unknown'
            success = result.get('success', False) if isinstance(result, dict) else False

            if success:
                await logger.info(f"JS execution successful: {result_type}")
            else:
                error = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'
                await logger.info(f"JS execution failed: {error}")

            return JavaScriptResult(**result)

        except asyncio.TimeoutError:
            await logger.info(f"JS execution timed out after {timeout_ms}ms")
            return JavaScriptResult(
                success=False,
                result_type="unserializable",
                error=f"Execution exceeded {timeout_ms}ms timeout",
                error_type="timeout",
            )
        except WebDriverException as e:
            await logger.info(f"JS execution WebDriver error: {e}")
            return JavaScriptResult(
                success=False,
                result_type="unserializable",
                error=str(e),
                error_type="execution",
            )
        except Exception as e:
            await logger.info(f"JS execution unexpected error: {e}")
            return JavaScriptResult(
                success=False,
                result_type="unserializable",
                error=str(e),
                error_type="execution",
            )


def _sync_cleanup(state: BrowserState) -> None:
    """Synchronous cleanup for signal handlers (runs in main thread)."""
    print("\n⚠ Signal received, cleaning up browser...", file=sys.stderr)
    if state.driver:
        try:
            state.driver.quit()
            print("✓ Browser closed", file=sys.stderr)
        except Exception as e:
            print(f"✗ Browser close error: {e}", file=sys.stderr)
    if state.mitmproxy_process:
        state.mitmproxy_process.terminate()
        try:
            state.mitmproxy_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            state.mitmproxy_process.kill()
    try:
        state.temp_dir.cleanup()
        state.capture_temp_dir.cleanup()
    except Exception:
        pass
    print("✓ Signal cleanup complete, exiting", file=sys.stderr)


@asynccontextmanager
async def lifespan(server_instance: FastMCP) -> typing.AsyncIterator[None]:
    """Manage browser lifecycle - initialization before requests, cleanup after shutdown."""
    state = await BrowserState.create()
    service = BrowserService(state)
    register_tools(service)

    # Register signal handlers to ensure cleanup on SIGTERM/SIGINT
    # This is critical for `claude mcp reconnect` which sends SIGTERM
    def signal_handler(signum: int, frame: typing.Any) -> None:
        _sync_cleanup(state)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("✓ Browser service initialized", file=sys.stderr)
    print(f"  Screenshot directory: {state.screenshot_dir}", file=sys.stderr)
    print(f"  Capture directory: {state.capture_dir}", file=sys.stderr)

    yield

    # SHUTDOWN: Cleanup after all requests complete (graceful shutdown path)
    if state.driver:
        await asyncio.to_thread(state.driver.quit)
    if state.mitmproxy_process:
        state.mitmproxy_process.terminate()
        try:
            state.mitmproxy_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            state.mitmproxy_process.kill()
    state.temp_dir.cleanup()
    state.capture_temp_dir.cleanup()
    print("✓ Server cleanup complete", file=sys.stderr)


mcp = FastMCP("selenium-browser-automation", lifespan=lifespan)


def main():
    """Entry point for uvx installation."""
    print("Starting Selenium Browser Automation MCP server", file=sys.stderr)
    print(
        "Note: This server uses CDP stealth injection to bypass bot detection",
        file=sys.stderr,
    )
    mcp.run()


if __name__ == "__main__":
    main()
