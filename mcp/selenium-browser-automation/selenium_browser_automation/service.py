from __future__ import annotations

__all__ = [
    'BrowserService',
]

import asyncio
import base64
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast
from urllib.parse import parse_qs, unquote, urlparse

import fastmcp.exceptions
import httpx
from cc_lib.schemas.base import SubsetModel
from cc_lib.types import JsonObject
from cc_lib.utils import Timer
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from . import chrome_profile_state_export
from .chrome_profiles import list_all_profiles
from .helpers import (
    ARIA_HIDDEN_REASON_KEYS,
    VISUAL_HIDDEN_REASON_KEYS,
    build_page_metadata,
    build_storage_init_script,
    count_tree_nodes,
    save_large_output_to_file,
)
from .models import (
    Browser,
    ChromeProfilesResult,
    ChromeProfileStateExportResult,
    ClearProxyResult,
    CLSMetric,
    ConfigureProxyResult,
    ConsoleLogEntry,
    ConsoleLogsResult,
    CoreWebVitals,
    DownloadResourceResult,
    FCPMetric,
    FocusableElement,
    HARExportResult,
    INPMetric,
    InteractiveElement,
    JavaScriptResult,
    LCPMetric,
    NavigationResult,
    NetworkCapture,
    NetworkRequest,
    PageTextResult,
    ProfileState,
    ProfileStateCookie,
    ProfileStateIndexedDB,
    ProfileStateOriginStorage,
    RequestTiming,
    ResizeWindowResult,
    SaveProfileStateResult,
    SetBlockedURLsResult,
    SleepResult,
    SlowestRequestSummary,
    SmartExtractionInfo,
    TTFBMetric,
    WaitForSelectorResult,
)
from .scripts import (
    ARIA_SNAPSHOT_SCRIPT,
    HOVER_OCCLUSION_SCRIPT,
    HOVER_STABILITY_SCRIPT,
    INDEXEDDB_CAPTURE_SCRIPT,
    INDEXEDDB_RESTORE_SCRIPT,
    NETWORK_MONITOR_CHECK_SCRIPT,
    NETWORK_MONITOR_SETUP_SCRIPT,
    RESOURCE_TIMING_SCRIPT,
    RESPONSE_BODY_CAPTURE_SCRIPT,
    TEXT_EXTRACTION_SCRIPT,
    VISUAL_TREE_SCRIPT,
    WEB_VITALS_SCRIPT,
    build_execute_javascript_async_script,
)
from .scroll import execute_scroll
from .state import BrowserState
from .tool_registry import ToolRegistry
from .tree_utils import compact_aria_tree, compact_visual_tree, serialize_aria_snapshot, serialize_visual_tree
from .validators import validate_css_selector as _validate_css_selector_impl  # used by _validate_css_selector wrapper

logger = logging.getLogger(__name__)


def _validate_css_selector(selector: str) -> None:
    """Validate CSS selector, converting ValueError to ToolError for MCP layer."""
    try:
        _validate_css_selector_impl(selector)
    except ValueError as e:
        raise fastmcp.exceptions.ToolError(str(e)) from None


VALID_URL_PREFIXES = ('http://', 'https://', 'file://', 'about:', 'data:', 'blob:')
LARGE_OUTPUT_THRESHOLD = 25_000


class BrowserService:
    """Browser automation service - protocol-agnostic, pure domain logic."""

    tool_registry: ClassVar[ToolRegistry] = ToolRegistry()

    def __init__(self, state: BrowserState) -> None:
        self.state = state  # Non-Optional - guaranteed by constructor

    async def close_browser(self) -> None:
        """Tear down browser and clear session state.

        Also clears origin tracking since a fresh browser is a fresh session.
        """
        if self.state.driver:
            await asyncio.to_thread(self.state.driver.quit)
            self.state.driver = None
            self.state.current_browser = None
        # Clear origin tracking and storage caches - new browser = new session
        self.state.origin_tracker.clear()
        self.state.local_storage_cache.clear()
        self.state.session_storage_cache.clear()
        self.state.indexed_db_cache.clear()
        # Clear lazy restore state - new browser = fresh session
        self.state.pending_profile_state = None
        self.state.restored_origins.clear()
        # Reset response body capture state - new browser needs interceptor reinstalled
        self.state.response_body_capture_enabled = False

    async def get_browser(self, enable_har_capture: bool = False, browser: Browser | None = None) -> webdriver.Chrome:
        """Initialize and return browser session (lazy singleton pattern).

        Args:
            enable_har_capture: Enable Chrome performance logging for HAR export.
            browser: Which browser to use - "chrome" or "chromium". Defaults to
                    the currently running browser, or "chromium" if none is running.
                    Use "chromium" to avoid AppleScript targeting conflicts when
                    your personal Chrome is running (different bundle ID).

        Returns:
            WebDriver instance
        """
        if self.state.driver is not None:
            return self.state.driver

        if browser is None:
            browser = self.state.current_browser or 'chromium'

        # CRITICAL: Stealth configuration to bypass Cloudflare bot detection
        opts = Options()

        # Set binary location for Chromium (macOS-specific for now)
        # TODO: Add cross-platform support (Linux: /usr/bin/chromium-browser, Windows: shutil.which)
        if browser == 'chromium':
            if sys.platform != 'darwin':
                raise fastmcp.exceptions.ToolError(
                    f'browser="chromium" is currently only supported on macOS (got {sys.platform})',
                )
            chromium_path = '/Applications/Chromium.app/Contents/MacOS/Chromium'
            if not Path(chromium_path).exists():
                raise fastmcp.exceptions.ToolError(
                    f'Chromium not found at {chromium_path}. Install with: brew install --cask chromium',
                )
            opts.binary_location = chromium_path
            logger.info('Using Chromium: %s', chromium_path)

        opts.add_argument('--disable-blink-features=AutomationControlled')
        opts.add_argument('--window-size=1920,1080')
        opts.add_experimental_option('excludeSwitches', ['enable-automation'])
        opts.add_experimental_option('useAutomationExtension', False)
        opts.add_experimental_option(
            'prefs',
            {
                'credentials_enable_service': False,
                'profile.password_manager_enabled': False,
            },
        )
        # Browser console logging (always enabled, lightweight)
        # Captures console.log/warn/error for debugging via get_console_logs()
        logging_prefs = {'browser': 'ALL'}

        # Performance logging for HAR export (opt-in due to overhead)
        # When enabled, Chrome continuously buffers Network.* events which adds
        # CPU, memory, and data transfer overhead even when export_har() isn't called.
        if enable_har_capture:
            logging_prefs['performance'] = 'ALL'
            opts.add_experimental_option('perfLoggingPrefs', {'enableNetwork': True})

        opts.set_capability('goog:loggingPrefs', logging_prefs)

        # Apply proxy configuration if mitmproxy is running
        # mitmproxy handles upstream authentication, Chrome just connects to local proxy
        if self.state.proxy_config and self.state.mitmproxy_process:
            opts.add_argument('--proxy-server=http://127.0.0.1:8080')
            opts.add_argument('--ignore-certificate-errors')  # mitmproxy uses self-signed certs
            logger.info(
                'Using local mitmproxy -> %s:%s', self.state.proxy_config['host'], self.state.proxy_config['port']
            )

        # Initialize driver in thread pool (blocking operation)
        self.state.driver = await asyncio.to_thread(webdriver.Chrome, options=opts)
        self.state.current_browser = browser

        # CRITICAL: CDP injection AFTER driver creation but BEFORE first navigation
        # This is what makes Selenium bypass Cloudflare where Playwright fails
        await asyncio.to_thread(
            self.state.driver.execute_cdp_cmd,
            'Page.addScriptToEvaluateOnNewDocument',
            {
                'source': """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    window.navigator.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """,
            },
        )

        return self.state.driver

    @tool_registry.register_tool
    async def get_page_text(
        self,
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
        driver = await self.get_browser()

        logger.info("Extracting text from '%s'", selector)

        # Execute the extraction script (loaded from scripts/)
        try:
            result = await asyncio.to_thread(driver.execute_script, TEXT_EXTRACTION_SCRIPT, selector)
        except WebDriverException as e:
            raise fastmcp.exceptions.ToolError(f'Failed to execute extraction script: {e}') from e

        # Handle selector not found
        if result.get('error'):
            raise fastmcp.exceptions.ToolError(
                f"Selector '{selector}' not found on page. "
                "Use get_aria_snapshot(selector='body') to discover valid selectors, "
                'or use get_page_html() to inspect the page structure.',
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
            text = '[No text content found in selected element]'

        # Get actual source element (may differ from selector in auto mode)
        source_element = result.get('sourceElement', selector)

        # Build smart_info only for auto mode
        smart_info = None
        if result.get('isSmartExtraction'):
            smart_info = SmartExtractionInfo(
                fallback_used=result.get('fallbackUsed', False),
                body_character_count=result.get('bodyCharacterCount', 0),
            )

        logger.info('Extracted %s characters from <%s>', f'{char_count:,}', source_element)

        # Save large text to file to preserve line structure
        saved_to_file = False
        file_path: str | None = None

        if char_count > LARGE_OUTPUT_THRESHOLD:
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
            filename = f'page_text_{timestamp}.txt'
            output_path = self.state.screenshot_dir / filename
            output_path.write_text(text, encoding='utf-8')

            saved_to_file = True
            file_path = str(output_path)
            text = text[:2000]  # Preview only

        return PageTextResult(
            title=title,
            url=url,
            source_element=source_element,
            text=text,
            character_count=char_count,
            saved_to_file=saved_to_file,
            file_path=file_path,
            smart_info=smart_info,
        )

    @tool_registry.register_tool
    async def get_page_html(self, selector: str | None = None, limit: int | None = None) -> str:
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
        driver = await self.get_browser()

        if selector:
            logger.info("Extracting HTML for '%s'%s", selector, f' (limit: {limit})' if limit else '')

            try:
                elements = await asyncio.to_thread(driver.find_elements, By.CSS_SELECTOR, selector)
            except WebDriverException as e:
                raise fastmcp.exceptions.ToolError(f"Invalid CSS selector '{selector}': {e}") from e

            count = len(elements)

            if count == 0:
                raise fastmcp.exceptions.ToolError(
                    f"No elements found matching selector '{selector}'. "
                    "Use get_aria_snapshot(selector='body') to discover valid selectors.",
                )

            actual_limit = min(count, limit) if limit is not None else count

            html_parts = []
            for i in range(actual_limit):
                try:
                    html = await asyncio.to_thread(driver.execute_script, 'return arguments[0].outerHTML;', elements[i])
                    html_parts.append(html)
                except WebDriverException:
                    continue  # Element may have become stale

            logger.info(
                'Extracted %s of %s elements%s',
                len(html_parts),
                count,
                f' (limited to {limit})' if limit and count > limit else '',
            )

            html_output = '\n'.join(html_parts)

            if len(html_output) > LARGE_OUTPUT_THRESHOLD:
                return save_large_output_to_file(
                    html_output,
                    self.state.screenshot_dir,
                    'page_html',
                    'html',
                )

            return html_output

        else:
            logger.info('Extracting full page HTML source')

            try:
                page_html: str = await asyncio.to_thread(lambda: driver.page_source)
            except WebDriverException as e:
                raise fastmcp.exceptions.ToolError(f'Failed to get page source: {e}') from e

            logger.info('Extracted %s characters of HTML', f'{len(page_html):,}')

            if len(page_html) > LARGE_OUTPUT_THRESHOLD:
                return save_large_output_to_file(
                    page_html,
                    self.state.screenshot_dir,
                    'page_html',
                    'html',
                )

            return page_html

    @tool_registry.register_tool
    async def screenshot(self, filename: str, full_page: bool = False) -> str:
        """Capture visual screenshot for verification and debugging.

        Use cases:
            - Verify visual appearance after taking actions
            - Debug layout or styling issues
            - Read text rendered in images (not in DOM)
            - Confirm visual state when structure is correct but appearance needs checking

        Args:
            filename: Output filename (e.g., "checkout-page.png")
            full_page: True for entire scrollable page, False for viewport only

        Returns:
            Absolute path to saved screenshot (use Read tool to view)

        Note: Requires vision processing. full_page=True uses CDP Page.captureScreenshot.
        """
        driver = await self.get_browser()

        screenshot_path = self.state.screenshot_dir / filename

        if full_page:
            logger.info('Taking full-page screenshot: %s', filename)
            # Use CDP to capture full page
            result = await asyncio.to_thread(
                driver.execute_cdp_cmd,
                'Page.captureScreenshot',
                {'captureBeyondViewport': True},
            )

            if 'data' not in result:
                raise fastmcp.exceptions.ToolError(
                    'CDP full-page capture returned no data. Use full_page=False for viewport screenshot.',
                )

            screenshot_data = base64.b64decode(result['data'])
            screenshot_path.write_bytes(screenshot_data)
            logger.info('Full-page screenshot saved to %s', screenshot_path)
            return str(screenshot_path)

        # Viewport screenshot
        logger.info('Taking viewport screenshot: %s', filename)
        await asyncio.to_thread(driver.save_screenshot, str(screenshot_path))
        logger.info('Screenshot saved to %s', screenshot_path)
        return str(screenshot_path)

    @tool_registry.register_tool
    async def download_resource(self, url: str, output_filename: str) -> DownloadResourceResult:
        """Download specific resource using current browser session's cookies and headers.

        Extracts User-Agent, cookies (with domain scoping), and Referer from the browser
        session to build browser-realistic requests. Critical for sites with bot detection
        or CDN hotlink protection — the CDN sees the request as coming from the same browser.

        PREREQUISITE: Call navigate() first to establish browser session.
        Without prior navigation, still works but may encounter bot detection.

        Args:
            url: Full URL to resource (http:// or https://) or file:// for local files.
            output_filename: Filename to save as (no path). Saved to screenshot temp dir.

        Returns:
            {'path': '/tmp/.../file.js', 'size_bytes': 26703, 'content_type': '...', 'status': 200, 'url': '...'}

        Errors: Raises ToolError if response status >= 400, network failure, or local file not found.
        """
        if not url.startswith(('http://', 'https://', 'file://')):
            raise fastmcp.exceptions.ValidationError('URL must start with http://, https://, or file://')

        safe_filename = ''.join(c if c.isalnum() or c in '.-_' else '_' for c in output_filename)
        if not safe_filename or safe_filename.startswith('.'):
            safe_filename = 'resource_' + safe_filename
        save_path = self.state.screenshot_dir / safe_filename

        if url.startswith('file://'):
            local_path = Path(unquote(urlparse(url).path))
            if not local_path.is_file():
                raise fastmcp.exceptions.ToolError(f'Local file not found: {local_path}')
            body = local_path.read_bytes()
            save_path.write_bytes(body)
            return DownloadResourceResult(
                path=str(save_path),
                size_bytes=len(body),
                content_type='application/octet-stream',
                status=200,
                url=url,
            )

        driver = await self.get_browser()
        logger.info('Downloading: %s', url)

        body, status, content_type = await _download_with_browser_context(self, driver, url)

        if status >= 400:
            raise fastmcp.exceptions.ToolError(f'Download failed with status {status}: {url}')

        save_path.write_bytes(body)
        logger.info('Downloaded %d bytes to %s', len(body), save_path)

        return DownloadResourceResult(
            path=str(save_path),
            size_bytes=len(body),
            content_type=content_type,
            status=status,
            url=url,
        )

    @tool_registry.register_tool
    async def get_aria_snapshot(
        self,
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
        driver = await self.get_browser()

        # Execute ARIA snapshot script (loaded from scripts/)
        result = await asyncio.to_thread(
            driver.execute_script,
            ARIA_SNAPSHOT_SCRIPT,
            selector,
            include_urls,
            include_hidden,
        )

        # Destructure: JS returns {tree, stats} or null (selector not found)
        if result is None:
            snapshot_data = None
            page_stats: dict[str, Any] = {}
        else:
            snapshot_data = result.get('tree')
            page_stats = result.get('stats', {})

        # Count nodes before compaction (for page metadata stats)
        raw_node_count = count_tree_nodes(snapshot_data)

        # Apply compaction if requested
        if compact_tree:
            snapshot_data = compact_aria_tree(snapshot_data)

        # Convert to Playwright-compatible YAML format using custom serializer
        yaml_output = serialize_aria_snapshot(snapshot_data)

        # Append page metadata footer
        metadata = build_page_metadata(
            page_stats=page_stats,
            include_page_info=include_page_info,
            include_urls=include_urls,
            compact_tree=compact_tree,
            raw_node_count=raw_node_count,
            compacted_node_count=count_tree_nodes(snapshot_data),
            hidden_reason_keys=ARIA_HIDDEN_REASON_KEYS,
        )
        if metadata:
            yaml_output += metadata

        if len(yaml_output) > LARGE_OUTPUT_THRESHOLD:
            return save_large_output_to_file(
                yaml_output,
                self.state.screenshot_dir,
                'aria_snapshot',
                'yaml',
            )

        return yaml_output

    @tool_registry.register_tool
    async def get_visual_tree(
        self,
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
        driver = await self.get_browser()

        # Execute visual tree script
        result = await asyncio.to_thread(
            driver.execute_script,
            VISUAL_TREE_SCRIPT,
            selector,
            include_urls,
            include_hidden,
        )

        # Destructure: JS returns {tree, stats} or null (selector not found)
        if result is None:
            return f'Selector not found: {selector}'
        snapshot_data = result.get('tree')
        page_stats: dict[str, Any] = result.get('stats', {})

        if snapshot_data is None:
            return f'Selector not found: {selector}'

        # Count nodes before compaction (for page metadata stats)
        raw_node_count = count_tree_nodes(snapshot_data)

        if compact_tree:
            snapshot_data = compact_visual_tree(snapshot_data)
            if snapshot_data is None:
                return ''

        # Serialize to YAML (same format as ARIA snapshot)
        yaml_output = serialize_visual_tree(snapshot_data)

        # Append page metadata footer
        metadata = build_page_metadata(
            page_stats=page_stats,
            include_page_info=include_page_info,
            include_urls=include_urls,
            compact_tree=compact_tree,
            raw_node_count=raw_node_count,
            compacted_node_count=count_tree_nodes(snapshot_data),
            hidden_reason_keys=VISUAL_HIDDEN_REASON_KEYS,
        )
        if metadata:
            yaml_output += metadata

        if len(yaml_output) > LARGE_OUTPUT_THRESHOLD:
            return save_large_output_to_file(
                yaml_output,
                self.state.screenshot_dir,
                'visual_tree',
                'yaml',
            )

        return yaml_output

    @tool_registry.register_tool
    async def get_interactive_elements(
        self,
        selector_scope: str,
        text_contains: str | None,
        tag_filter: Sequence[str] | None,
        limit: int | None,
    ) -> Sequence[InteractiveElement]:
        """Find clickable elements by text or other filters. Returns CSS selectors for click().

        This is the tool for finding elements by text content.
        Example: get_interactive_elements(text_contains="Continue") → returns CSS selector for click()

        Args:
            selector_scope: CSS selector to limit search (e.g., ".wizard" or "body")
            text_contains: Filter by text content (case-insensitive, None = no filter)
            tag_filter: Only specific tags (e.g., ["button", "a"], None = all tags)
            limit: Max results to return (None = unlimited)

        Returns:
            list[InteractiveElement]: Filtered interactive elements with selectors for clicking

        Detection Methods:
            Elements are detected as interactive if ANY of these conditions are true:
            - CSS cursor:pointer
            - Native <button>, <a>, <input>, <select>, <textarea> elements
            - Elements with role="button" or role="link"
            - Elements with onclick property (native handler)
            - Elements with React __reactProps containing onClick function

        Known Limitations (what we CANNOT detect):
            - addEventListener handlers (not exposed as element properties)
            - Event delegation (parent element handles clicks for children)
            - Vue/Angular/Svelte handlers (framework-specific, often hidden in production)
            - CSS-only interactions (checkbox hack, :target navigation)
            - Elements that become interactive after state changes
            - Touch-only handlers (ontouchstart without onclick)

        When detection fails, try:
            1. Use get_aria_snapshot() to understand the page structure
            2. Look for nearby buttons or the CHEVRON_RIGHT pattern
            3. Scope search with selector_scope to reduce false positives
        """
        driver = await self.get_browser()

        logger.info('Finding interactive elements in scope: %s', selector_scope)

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

                // Check for click handlers (native onclick and React onClick)
                const hasOnclick = el.onclick !== null;
                const hasReactOnClick = Object.keys(el).some(k => {
                    if (!k.startsWith('__reactProps')) return false;
                    const props = el[k];
                    return props && typeof props.onClick === 'function';
                });

                const isInteractive = hasPointer || isButton || isLink || isInput || hasOnclick || hasReactOnClick;

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
                    classes: el.getAttribute('class') || ''
                });
            });

            return results;
            """,
            {
                'scopeSelector': selector_scope,
                'textFilter': text_filter_lower,
                'tagFilter': tag_filter_upper,
                'maxLimit': limit,
            },
        )

        logger.info('Found %s interactive elements (filtered)', len(elements))
        return [InteractiveElement(**el) for el in elements]

    @tool_registry.register_tool
    async def get_focusable_elements(self, only_tabbable: bool) -> Sequence[FocusableElement]:
        """Get keyboard-navigable elements sorted by tab order.

        Args:
            only_tabbable: True = Tab key only (tabindex >= 0)
                          False = includes programmatic focus (tabindex >= -1)

        Returns:
            list[FocusableElement]: Sorted by tab order, each with tag, text, selector, tab_index
        """
        driver = await self.get_browser()

        logger.info('Finding focusable elements (only_tabbable=%s)', only_tabbable)

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
                        classes: el.getAttribute('class') || ''
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
            {'minTabIndex': min_tab_index},
        )

        logger.info('Found %s focusable elements', len(elements))
        return [FocusableElement(**el) for el in elements]

    @tool_registry.register_tool
    async def click(
        self,
        css_selector: str,
        wait_for_network: bool = False,
        network_timeout: int = 10000,
    ) -> None:
        """Click an element. Auto-waits for visibility and clickability.

        Accepts standard CSS selectors only. To find elements by text content,
        use get_interactive_elements(text_contains='...') which returns CSS selectors.

        Args:
            css_selector: Standard CSS selector from get_interactive_elements()
            wait_for_network: Add fixed delay after click (use wait_for_network_idle() instead)
            network_timeout: Delay duration in ms

        Workflow:
            1. get_interactive_elements(text_contains='Submit') - find element by text
            2. click(css_selector) - click using the CSS selector it returned
            3. wait_for_network_idle() - wait for dynamic content
            4. get_aria_snapshot() - understand new page state
        """
        driver = await self.get_browser()

        # PRE-ACTION: Capture localStorage before click (might navigate away)
        await _capture_current_origin_storage(self, driver)
        url_before = driver.current_url

        logger.info('Clicking element: %s%s', css_selector, ' (with delay)' if wait_for_network else '')

        _validate_css_selector(css_selector)

        # Wait for element to be clickable (up to 10 seconds)
        try:
            element = await asyncio.to_thread(
                WebDriverWait(driver, 10).until,
                EC.element_to_be_clickable((By.CSS_SELECTOR, css_selector)),
            )
        except WebDriverException as e:
            raise fastmcp.exceptions.ToolError(
                f"Failed to find clickable element '{css_selector}': {e}\n"
                f"Use get_interactive_elements(text_contains='...') to discover valid selectors.",
            ) from None

        # Click element
        try:
            await asyncio.to_thread(element.click)
        except WebDriverException as e:
            error_msg = str(e).lower()
            if 'intercepted' in error_msg or 'other element would receive' in error_msg:
                raise fastmcp.exceptions.ToolError(
                    f"Element '{css_selector}' is obscured by another element. "
                    f'Try: close overlays/modals, scroll into view, or use '
                    f'execute_javascript("document.querySelector(\'{css_selector}\').click()")',
                ) from None
            raise fastmcp.exceptions.ToolError(f"Click failed on '{css_selector}': {e}") from None

        if wait_for_network:
            delay_sec = network_timeout / 1000
            logger.info('Waiting %ss for content to load', delay_sec)
            await asyncio.sleep(delay_sec)
            logger.info('Delay complete')

        # POST-ACTION: Check if navigation occurred, track new origin and restore localStorage
        url_after = driver.current_url
        if url_after != url_before:
            self.state.origin_tracker.add_origin(url_after)
            logger.info('Navigation detected: %s -> %s', url_before, url_after)
            # Lazy restore: if we navigated to a new origin with pending storage state
            await _restore_pending_profile_state_for_current_origin(self, driver)

        logger.info('Click successful')

    @tool_registry.register_tool
    async def wait_for_network_idle(self, timeout: int = 10000) -> None:
        """Wait for network activity to settle after clicks or dynamic content loads.

        Uses JavaScript instrumentation to monitor Fetch and XMLHttpRequest activity.
        Waits for no active requests and 500ms of idle time.

        Args:
            timeout: Timeout in milliseconds (default 10000ms)

        Note: Uses JavaScript instrumentation of Fetch/XHR APIs to track network activity.
              Waits for no active requests + 500ms idle threshold.
        """
        driver = await self.get_browser()

        # Step 1: Inject monitoring script (loaded from scripts/)
        await asyncio.to_thread(driver.execute_script, NETWORK_MONITOR_SETUP_SCRIPT)
        logger.info('Network monitor injected')

        # Step 2: Poll for idle state (500ms threshold)
        start_time = time.time()
        idle_threshold_ms = 500
        timeout_s = timeout / 1000

        while time.time() - start_time < timeout_s:
            status = await asyncio.to_thread(driver.execute_script, NETWORK_MONITOR_CHECK_SCRIPT)

            if status['activeRequests'] == 0:
                if status['lastRequestTime'] is None:
                    # No requests made yet - check elapsed time since monitoring started
                    elapsed = time.time() - start_time
                    if elapsed >= idle_threshold_ms / 1000:
                        logger.info('Network idle (no requests made)')
                        return
                else:
                    # Check time since last request completed
                    elapsed_since_last = status['currentTime'] - status['lastRequestTime']
                    if elapsed_since_last >= idle_threshold_ms:
                        logger.info('Network idle after %.2fs', elapsed_since_last / 1000)
                        return

            await asyncio.sleep(0.05)  # Poll every 50ms

        logger.info('Network idle timeout after %ss', timeout_s)

    @tool_registry.register_tool
    async def press_key(self, key: str) -> None:
        """Press a keyboard key or key combination.

        Args:
            key: Key name or combination. Common keys:
                - Single keys: 'ESCAPE', 'ENTER', 'TAB', 'BACKSPACE', 'DELETE', 'ARROW_UP', 'ARROW_DOWN', 'ARROW_LEFT', 'ARROW_RIGHT'
                - Special keys: 'F1' through 'F12', 'HOME', 'END', 'PAGE_UP', 'PAGE_DOWN'
                - Modifiers: Use + for combinations like 'CONTROL+A', 'META+V'

        Examples:
            - press_key('ESCAPE') - Close modals
            - press_key('ENTER') - Submit forms
            - press_key('TAB') - Navigate between fields
            - press_key('CONTROL+A') - Select all
            - press_key('META+V') - Paste (Cmd+V on Mac)

        Note: Key names use Selenium's Keys enum (uppercase with underscores).
              For combinations, use + (e.g., 'CONTROL+A').
        """
        driver = await self.get_browser()

        # PRE-ACTION: Capture localStorage before key press (ENTER might submit form and navigate)
        await _capture_current_origin_storage(self, driver)
        url_before = driver.current_url

        logger.info('Pressing key: %s', key)

        # Map common key names to Selenium Keys
        # Handle key combinations (e.g., "CONTROL+A")
        if '+' in key:
            parts = key.split('+')
            keys_combo = []
            for part in parts:
                key_attr = part.upper().replace(' ', '_')
                if hasattr(Keys, key_attr):
                    keys_combo.append(getattr(Keys, key_attr))
                else:
                    keys_combo.append(part)

            # Send key combination
            body = await asyncio.to_thread(driver.find_element, By.TAG_NAME, 'body')
            await asyncio.to_thread(body.send_keys, *keys_combo)
        else:
            # Single key
            key_attr = key.upper().replace(' ', '_')
            if hasattr(Keys, key_attr):
                key_value = getattr(Keys, key_attr)
            else:
                key_value = key

            body = await asyncio.to_thread(driver.find_element, By.TAG_NAME, 'body')
            await asyncio.to_thread(body.send_keys, key_value)

        # POST-ACTION: Check if navigation occurred, track new origin and restore localStorage
        url_after = driver.current_url
        if url_after != url_before:
            self.state.origin_tracker.add_origin(url_after)
            logger.info('Navigation detected: %s -> %s', url_before, url_after)
            # Lazy restore: if we navigated to a new origin with pending storage state
            await _restore_pending_profile_state_for_current_origin(self, driver)

        logger.info('Key press successful')

    @tool_registry.register_tool
    async def type_text(self, text: str, delay_ms: int = 0) -> None:
        """Type text character by character with optional delay between keystrokes.

        Args:
            text: Text to type
            delay_ms: Optional delay between keystrokes in milliseconds (default 0)

        Note: For simple form filling, prefer using element.send_keys() directly as it's faster.
              Use this tool when you need to simulate human typing or trigger per-character keyboard events.

        Example:
            - type_text('Hello, world!') - Type text instantly
            - type_text('search query', delay_ms=50) - Type with 50ms delay between chars
        """
        driver = await self.get_browser()

        logger.info('Typing text: "%s"%s', text, f' with {delay_ms}ms delay' if delay_ms > 0 else '')

        active_element = await asyncio.to_thread(lambda: driver.switch_to.active_element)

        if delay_ms > 0:
            # Type with delay between characters
            for char in text:
                await asyncio.to_thread(active_element.send_keys, char)
                await asyncio.sleep(delay_ms / 1000)
        else:
            # Type all at once
            await asyncio.to_thread(active_element.send_keys, text)

        logger.info('Text typing successful')

    @tool_registry.register_tool
    async def hover(self, css_selector: str, duration_ms: int = 0) -> None:
        """Move mouse over an element to trigger hover states.

        Essential for dropdown menus, tooltips, and hover-triggered UI.
        JavaScript events (mouseover/mouseenter) don't trigger CSS :hover -
        this tool uses real mouse simulation via ActionChains.

        Accepts standard CSS selectors only. To find elements by text content,
        use get_interactive_elements(text_contains='...') which returns CSS selectors.

        Args:
            css_selector: Standard CSS selector from get_interactive_elements()
            duration_ms: Hold duration in ms (for menus that need sustained hover)

        Workflow:
            1. get_interactive_elements(text_contains='Products') - find menu trigger
            2. hover(css_selector) - reveal dropdown
            3. get_aria_snapshot() - see dropdown content
            4. click(dropdown_item_selector) - select item
        """
        driver = await self.get_browser()

        # Validate duration
        if duration_ms < 0:
            raise ValueError('duration_ms cannot be negative')
        if duration_ms > 30000:
            raise ValueError('duration_ms exceeds maximum of 30000ms (30 seconds)')

        logger.info('Hovering over element: %s', css_selector)

        _validate_css_selector(css_selector)

        # Wait for element to be present
        try:
            element = await asyncio.to_thread(
                WebDriverWait(driver, 10).until,
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
            )
        except WebDriverException as e:
            raise fastmcp.exceptions.ToolError(
                f"Failed to find element for hover '{css_selector}': {e}\n"
                f"Use get_interactive_elements(text_contains='...') to discover valid selectors.",
            ) from None

        # Scroll element into view (Playwright does this automatically)
        await asyncio.to_thread(
            driver.execute_script,
            "arguments[0].scrollIntoView({behavior: 'instant', block: 'nearest'});",
            element,
        )

        # Verify element is visible (not display:none, visibility:hidden, etc.)
        is_displayed = await asyncio.to_thread(element.is_displayed)
        if not is_displayed:
            raise ValueError(f"Element '{css_selector}' is not visible - cannot hover")

        # Multi-signal stability check (based on Playwright's approach)
        # Uses: requestAnimationFrame timing, getAnimations() API, distance threshold
        stability_result = await asyncio.to_thread(
            driver.execute_script,
            HOVER_STABILITY_SCRIPT,
            element,
        )

        # Log stability check results
        if stability_result.get('hasInfiniteAnimation'):
            logger.warning('Element has infinite animation - hover may be inconsistent')

        if not stability_result.get('stable'):
            if stability_result.get('runningAnimations', 0) > 0:
                logger.info(
                    'Element has %s running animation(s), proceeding after %s frame checks',
                    stability_result['runningAnimations'],
                    stability_result['framesChecked'],
                )
            else:
                logger.info(
                    'Element did not stabilize after %s frames (final distance: %.1fpx)',
                    stability_result['framesChecked'],
                    stability_result.get('finalDistance', 0),
                )

        # Verify element receives pointer events (not obscured by overlay/modal)
        pointer_check = await asyncio.to_thread(
            driver.execute_script,
            HOVER_OCCLUSION_SCRIPT,
            element,
        )
        if pointer_check == 'obscured':
            raise ValueError(
                f"Element '{css_selector}' is obscured by another element at its center. "
                'A modal, overlay, or other element may be blocking it.',
            )

        # Move mouse to element center
        actions = ActionChains(driver)
        await asyncio.to_thread(actions.move_to_element(element).perform)

        # Hold if duration specified (for menus that need sustained hover)
        if duration_ms > 0:
            logger.info('Holding hover for %sms', duration_ms)
            await asyncio.sleep(duration_ms / 1000)

        logger.info('Hover successful')

    @tool_registry.register_tool
    async def scroll(
        self,
        direction: Literal['up', 'down', 'left', 'right'] | None = None,
        scroll_amount: int = 3,
        css_selector: str | None = None,
        behavior: Literal['instant', 'smooth'] = 'instant',
        position: Literal['top', 'bottom', 'left', 'right'] | None = None,
    ) -> JsonObject:
        """Scroll the page, a container, or an element into view.

        **Relative scrolling** (direction parameter — scrollBy):
        - direction only: Scroll viewport by amount
        - direction + css_selector: Scroll within a container by amount

        **Absolute scrolling** (position parameter — scrollTo):
        - position only: Scroll viewport to an edge (top/bottom/left/right)
        - position + css_selector: Scroll container to an edge

        **Scroll into view** (css_selector only):
        - Scrolls until the target element is centered in the viewport

        Args:
            direction: Scroll direction: 'up', 'down', 'left', 'right'.
                Required for relative scrolling. Mutually exclusive with position.
            scroll_amount: Number of ticks to scroll (1-20, default 3).
                Each tick produces 100 CSS pixels of scroll delta. Only used with direction.
            css_selector: CSS selector for either:
                - Target element to scroll into view (when alone)
                - Container to scroll within (when combined with direction or position)
            behavior: Scroll animation: 'instant' (default) or 'smooth'.
                'instant' scrolls immediately with no animation.
                'smooth' animates using the browser's native smooth scroll (~300-500ms)
                and waits for the animation to complete before returning position data.
            position: Absolute scroll target: 'top', 'bottom', 'left', 'right'.
                Scrolls to the edge of the page or container. Mutually exclusive with direction.

        Returns:
            Dict with scroll results including mode, scroll position, page dimensions,
            and a 'scrolled' boolean indicating whether the scroll position changed.

        Examples:
            - scroll(direction='down') - Scroll viewport down ~300px
            - scroll(direction='down', scroll_amount=10) - Scroll down ~1000px
            - scroll(css_selector='#footer') - Scroll #footer into view (centered)
            - scroll(direction='down', css_selector='.chat-log') - Scroll within .chat-log
            - scroll(direction='down', behavior='smooth') - Smooth animated scroll
            - scroll(position='bottom') - Scroll to bottom of page
            - scroll(position='top') - Scroll to top of page
            - scroll(position='right', css_selector='#data-table') - Scroll to rightmost column
            - scroll(position='bottom', css_selector='.chat-log') - Scroll chat to end
            - scroll(position='bottom', behavior='smooth') - Smooth scroll to bottom

        Notes:
            - CSS scroll-snap may cause actual scroll position to differ from requested amount.
            - Use the returned scroll position and 'scrolled' boolean to detect boundaries.
            - For iframe content, switch to the iframe context first.

        Workflow:
            1. get_aria_snapshot('body') - understand page structure
            2. scroll(direction='down') - scroll to see more content
            3. get_aria_snapshot('body') - see updated content
        """
        driver = await self.get_browser()
        logger.info(
            'Scroll: direction=%s, position=%s, selector=%s, amount=%s, behavior=%s',
            direction,
            position,
            css_selector,
            scroll_amount,
            behavior,
        )
        try:
            result = await asyncio.to_thread(
                execute_scroll,
                driver,
                direction=direction,
                scroll_amount=scroll_amount,
                css_selector=css_selector,
                behavior=behavior,
                position=position,
            )
        except ValueError as e:
            raise fastmcp.exceptions.ToolError(str(e)) from None
        logger.info('Scroll complete: %s, scrolled=%s', result.get('mode'), result.get('scrolled'))
        return result

    @tool_registry.register_tool
    async def sleep(self, duration_ms: int, reason: str | None = None) -> SleepResult:
        """Pause execution for a fixed duration. Use sparingly.

        This is a simple time-based delay. For most automation scenarios,
        prefer condition-based waits instead:
        - wait_for_selector() - Wait for specific UI elements
        - wait_for_network_idle() - Wait for network activity to settle

        Use sleep() only when you need a fixed delay for:
        - CSS animations with known duration
        - Debounce timers
        - Rate limiting between actions
        - Timing-sensitive test scenarios

        Args:
            duration_ms: Sleep duration in milliseconds (max 300000 = 5 min)
            reason: Optional context for logging (e.g., "CSS animation")

        Returns:
            Dict with slept_ms confirming the sleep duration
        """

        # Validation
        if duration_ms < 0:
            raise ValueError('duration_ms cannot be negative')
        if duration_ms > 300000:
            raise ValueError(
                'duration_ms exceeds maximum of 300000ms (5 minutes). '
                'Such long delays usually indicate missing condition-based waiting logic. '
                'Consider using wait_for_selector() or wait_for_network_idle() instead.',
            )

        # Graduated warnings for AI agents
        if duration_ms > 10000:
            logger.info(
                'Warning: Long sleep(%sms) requested. '
                'Consider wait_for_selector() or wait_for_network_idle() for dynamic content.',
                duration_ms,
            )

        if reason:
            logger.info('Sleeping %sms: %s', duration_ms, reason)
        else:
            logger.info('Sleeping %sms', duration_ms)

        await asyncio.sleep(duration_ms / 1000)

        return SleepResult(slept_ms=duration_ms)

    @tool_registry.register_tool
    async def wait_for_selector(
        self,
        css_selector: str,
        state: Literal['visible', 'hidden', 'attached', 'detached'] = 'visible',
        timeout: int = 30000,
    ) -> WaitForSelectorResult:
        """Wait for an element matching the selector to reach a desired state.

        More reliable than wait_for_network_idle() for modern SPAs because it waits
        for specific UI state rather than network activity. Use this when you need
        to interact with a specific element after dynamic content loads.

        Accepts standard CSS selectors only.

        Args:
            css_selector: Standard CSS selector to wait for
            state: Target state (default "visible"):
                - "visible": Element in DOM AND displayed (not display:none/visibility:hidden)
                - "hidden": Element not visible OR not in DOM
                - "attached": Element present in DOM (regardless of visibility)
                - "detached": Element removed from DOM
            timeout: Maximum wait in milliseconds (default 30000, max 300000)

        Returns:
            Dict with selector, state achieved, and elapsed_ms

        Examples:
            - wait_for_selector("#modal")  # Wait for modal to appear
            - wait_for_selector(".loading", state="hidden")  # Wait for loader to disappear
            - wait_for_selector("[data-loaded]", state="attached")  # Wait for attribute

        Raises:
            ValueError: If timeout is invalid or selector is empty
            TimeoutError: If element doesn't reach desired state within timeout
        """
        driver = await self.get_browser()

        # Validation
        if not css_selector or not css_selector.strip():
            raise ValueError('css_selector cannot be empty')
        if timeout < 0:
            raise ValueError('timeout cannot be negative')
        if timeout > 300000:
            raise ValueError('timeout exceeds maximum of 300000ms (5 minutes)')

        logger.info("Waiting for selector '%s' to be %s", css_selector, state)

        _validate_css_selector(css_selector)

        # Early validation: detect invalid CSS before entering polling loop.
        # Without this, an invalid selector silently wastes the full timeout.
        try:
            await asyncio.to_thread(driver.find_elements, By.CSS_SELECTOR, css_selector)
        except WebDriverException as e:
            raise fastmcp.exceptions.ToolError(
                f"Invalid CSS selector '{css_selector}': {e}\n"
                f'Use get_aria_snapshot() to discover valid selectors, or '
                f'get_interactive_elements() to find elements by text.',
            ) from None

        start_time = time.time()
        timeout_s = timeout / 1000
        polling_interval = 0.05  # 50ms polling

        while time.time() - start_time < timeout_s:
            try:
                elements = await asyncio.to_thread(driver.find_elements, By.CSS_SELECTOR, css_selector)

                if state == 'attached':
                    # Element exists in DOM
                    if elements:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        logger.info("Selector '%s' attached after %sms", css_selector, elapsed_ms)
                        return WaitForSelectorResult(
                            selector=css_selector,
                            state='attached',
                            elapsed_ms=elapsed_ms,
                            element_count=len(elements),
                        )

                elif state == 'detached':
                    # Element removed from DOM
                    if not elements:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        logger.info("Selector '%s' detached after %sms", css_selector, elapsed_ms)
                        return WaitForSelectorResult(
                            selector=css_selector,
                            state='detached',
                            elapsed_ms=elapsed_ms,
                        )

                elif state == 'visible':
                    # Element exists AND is displayed
                    for element in elements:
                        is_displayed = await asyncio.to_thread(element.is_displayed)
                        if is_displayed:
                            elapsed_ms = int((time.time() - start_time) * 1000)
                            logger.info("Selector '%s' visible after %sms", css_selector, elapsed_ms)
                            return WaitForSelectorResult(
                                selector=css_selector,
                                state='visible',
                                elapsed_ms=elapsed_ms,
                                element_count=len(elements),
                            )

                elif state == 'hidden':
                    # Element not in DOM OR not displayed
                    if not elements:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        logger.info("Selector '%s' hidden (not in DOM) after %sms", css_selector, elapsed_ms)
                        return WaitForSelectorResult(
                            selector=css_selector,
                            state='hidden',
                            elapsed_ms=elapsed_ms,
                            reason='not_in_dom',
                        )
                    # Check if all matching elements are hidden
                    all_hidden = True
                    for element in elements:
                        is_displayed = await asyncio.to_thread(element.is_displayed)
                        if is_displayed:
                            all_hidden = False
                            break
                    if all_hidden:
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        logger.info("Selector '%s' hidden (not displayed) after %sms", css_selector, elapsed_ms)
                        return WaitForSelectorResult(
                            selector=css_selector,
                            state='hidden',
                            elapsed_ms=elapsed_ms,
                            reason='not_displayed',
                            element_count=len(elements),
                        )

            except Exception:  # exception_safety_linter.py: swallowed-exception — polling resilience; selector evaluation may hit stale element
                pass

            await asyncio.sleep(polling_interval)

        # Timeout reached
        elapsed_ms = int((time.time() - start_time) * 1000)

        # Build helpful error message for AI agents
        try:
            elements = await asyncio.to_thread(driver.find_elements, By.CSS_SELECTOR, css_selector)
            element_count = len(elements)
            if elements:
                first_displayed = await asyncio.to_thread(elements[0].is_displayed)
                current_state = 'visible' if first_displayed else 'in DOM but hidden'
            else:
                current_state = 'not in DOM'
        except (
            Exception
        ):  # exception_safety_linter.py: swallowed-exception — best-effort diagnostics for timeout error message
            element_count = 0
            current_state = 'unknown (selector may be invalid)'

        raise TimeoutError(
            f"wait_for_selector('{css_selector}', state='{state}') timed out after {elapsed_ms}ms. "
            f'Current state: {current_state} (found {element_count} element(s)). '
            f'Possible causes: (1) Selector is incorrect, (2) Element is in iframe, '
            f"(3) Element requires user action to appear, (4) Page didn't finish loading. "
            f'Try: verify selector with get_page_html(), check for iframes, or increase timeout.',
        )

    @tool_registry.register_tool
    async def list_chrome_profiles(
        self,
        verbose: bool = False,
    ) -> ChromeProfilesResult:
        """List all Chrome profiles with metadata.

        Args:
            verbose: If True, include all metadata fields.
                     If False (default), show only essential fields.

        Returns:
            ChromeProfilesResult with profiles list, count, and default profile

        Example (default):
            Returns essential fields: name, email, profile_dir, etc.

        Example (verbose):
            Returns all fields including avatar settings, creation time, etc.
        """
        logger.info('Listing Chrome profiles (verbose=%s)', verbose)

        # Call module function
        result = await asyncio.to_thread(list_all_profiles, verbose=verbose)

        logger.info('Found %s profiles', result.total_count)

        return result

    @tool_registry.register_tool
    async def resize_window(self, width: int, height: int) -> ResizeWindowResult:
        """Resize the browser window to specified dimensions.

        Useful for responsive design testing and mobile simulation.

        Args:
            width: Window width in pixels
            height: Window height in pixels

        Returns:
            Dict with actual width and height after resize

        Common presets:
            - Mobile (iPhone SE): 375 x 667
            - Tablet (iPad): 768 x 1024
            - Desktop (1080p): 1920 x 1080
            - Desktop (1440p): 2560 x 1440

        Example:
            resize_window(375, 667)  # Mobile viewport
            screenshot("mobile-view.png")

        Note:
            Actual window size may differ from requested due to OS constraints
            (e.g., macOS enforces minimum ~500px width). The returned dimensions
            reflect the actual size achieved.
        """
        driver = await self.get_browser()

        # Validation: positive integers only
        if width <= 0 or height <= 0:
            raise ValueError(f'Width and height must be positive integers. Got: {width}x{height}')

        logger.info('Resizing window to %sx%s', width, height)

        await asyncio.to_thread(driver.set_window_size, width, height)

        # Get actual size (may differ due to OS constraints)
        size = await asyncio.to_thread(driver.get_window_size)

        logger.info('Window resized to %sx%s', size['width'], size['height'])

        return ResizeWindowResult(width=size['width'], height=size['height'])

    @tool_registry.register_tool
    async def capture_web_vitals(self, timeout_ms: int = 5000) -> CoreWebVitals:
        """Capture Core Web Vitals (LCP, CLS, INP, FCP, TTFB) for current page.

        Uses JavaScript Performance APIs to collect metrics post-navigation.
        Must be called after navigate().

        Args:
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
        driver = await self.get_browser()

        start_time = time.time()
        current_url = driver.current_url

        logger.info('Capturing Core Web Vitals for %s', current_url)

        errors: list[str] = []

        # Use execute_async_script for Promise-based collection (script from scripts/)
        results = await asyncio.to_thread(
            driver.execute_async_script,
            WEB_VITALS_SCRIPT,
            timeout_ms,
        )

        collection_duration = (time.time() - start_time) * 1000

        # Parse results into Pydantic models
        T = TypeVar('T', bound=SubsetModel)

        def parse_metric(data: JsonObject | None, model_cls: type[T]) -> T | None:
            if not data:
                return None
            if 'error' in data:
                errors.append(f'{data.get("name", "Unknown")}: {data["error"]}')
                return None
            return model_cls(**data)

        vitals = CoreWebVitals(
            url=current_url,
            timestamp=time.time(),
            fcp=parse_metric(results.get('fcp') if results else None, FCPMetric),
            lcp=parse_metric(results.get('lcp') if results else None, LCPMetric),
            ttfb=parse_metric(results.get('ttfb') if results else None, TTFBMetric),
            cls=parse_metric(results.get('cls') if results else None, CLSMetric),
            inp=parse_metric(results.get('inp') if results else None, INPMetric),
            collection_duration_ms=collection_duration,
            errors=errors,
        )

        # Log summary
        metrics_found = []
        if vitals.fcp:
            metrics_found.append(f'FCP={vitals.fcp.value:.0f}ms ({vitals.fcp.rating})')
        if vitals.lcp:
            metrics_found.append(f'LCP={vitals.lcp.value:.0f}ms ({vitals.lcp.rating})')
        if vitals.ttfb:
            metrics_found.append(f'TTFB={vitals.ttfb.value:.0f}ms ({vitals.ttfb.rating})')
        if vitals.cls:
            metrics_found.append(f'CLS={vitals.cls.value:.3f} ({vitals.cls.rating})')
        if vitals.inp:
            metrics_found.append(f'INP={vitals.inp.value:.0f}ms ({vitals.inp.rating})')

        logger.info('Web Vitals captured in %.0fms: %s', collection_duration, ', '.join(metrics_found) or 'none')

        return vitals

    @tool_registry.register_tool
    async def get_resource_timings(
        self,
        clear_resource_timing_buffer: bool = False,
        min_duration_ms: int = 0,
    ) -> NetworkCapture:
        driver = await self.get_browser()

        start_time = time.time()
        current_url = driver.current_url

        logger.info('Getting resource timings for %s', current_url)

        # Collect resource timing via JavaScript Performance API (script from scripts/)
        raw_entries = await asyncio.to_thread(driver.execute_script, RESOURCE_TIMING_SCRIPT)

        # Handle empty/null response
        if not raw_entries:
            raw_entries = []

        # Filter by min_duration_ms
        if min_duration_ms > 0:
            raw_entries = [e for e in raw_entries if e.get('duration_ms', 0) >= min_duration_ms]

        # Convert to NetworkRequest models
        requests = []
        total_size = 0
        for i, entry in enumerate(raw_entries):
            timing_data = entry.get('timing', {})
            req = NetworkRequest(
                request_id=str(i),
                url=entry.get('url', ''),
                method=entry.get('method', 'GET'),
                resource_type=entry.get('resource_type'),
                encoded_data_length=entry.get('encoded_data_length', 0),
                started_at=entry.get('started_at', 0),
                duration_ms=entry.get('duration_ms'),
                timing=RequestTiming(
                    blocked=timing_data.get('blocked', 0),
                    dns=timing_data.get('dns', 0),
                    connect=timing_data.get('connect', 0),
                    ssl=timing_data.get('ssl', 0),
                    send=timing_data.get('send', 0),
                    wait=timing_data.get('wait', 0),
                    receive=timing_data.get('receive', 0),
                )
                if timing_data
                else None,
            )
            requests.append(req)
            total_size += entry.get('encoded_data_length', 0)

        # Sort by duration (slowest first) for summary
        sorted_by_duration = sorted(requests, key=lambda r: r.duration_ms or 0, reverse=True)
        slowest = [
            SlowestRequestSummary(url=r.url, duration_ms=r.duration_ms, type=r.resource_type)
            for r in sorted_by_duration[:10]
        ]

        # Count by resource type
        type_counts: dict[str, int] = {}
        for r in requests:
            t = r.resource_type or 'other'
            type_counts[t] = type_counts.get(t, 0) + 1

        # Calculate total time (longest request duration)
        total_time = max((r.duration_ms or 0) for r in requests) if requests else 0

        # Clear resource timing buffer if requested
        if clear_resource_timing_buffer:
            await asyncio.to_thread(
                driver.execute_script,
                'performance.clearResourceTimings();',
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
            slowest_url = slowest[0].url
            # Truncate URL for logging
            if len(slowest_url) > 60:
                slowest_url = slowest_url[:57] + '...'
            logger.info(
                'Captured %s requests in %.0fms, slowest: %.0fms (%s)',
                len(requests),
                collection_duration,
                slowest[0].duration_ms or 0,
                slowest_url,
            )
        else:
            logger.info('Captured %s requests in %.0fms', len(requests), collection_duration)

        return result

    @tool_registry.register_tool
    async def get_console_logs(
        self,
        level_filter: Literal['ALL', 'SEVERE', 'WARNING', 'INFO'] | None = None,
        pattern: str | None = None,
    ) -> ConsoleLogsResult:
        """Get browser console logs (console.log, console.error, etc.).

        Retrieves JavaScript console output for debugging. Logs are cleared after
        each retrieval, so call this to capture logs between page interactions.

        Args:
            level_filter: Only include logs at or above this level.
                         SEVERE = errors only, WARNING = warnings+errors,
                         INFO = all (including console.log), ALL = everything.
                         Default None = ALL.
            pattern: Regex pattern to filter messages (case-insensitive).
                    Example: "error|failed" to find error-related messages.

        Returns:
            ConsoleLogsResult with logs array and counts by level.

        Log levels (Chrome's naming):
            - SEVERE: console.error(), uncaught exceptions
            - WARNING: console.warn()
            - INFO: console.log(), console.info(), console.debug()

        Notes:
            - Console logging is always enabled (no special setup needed)
            - Each call clears the log buffer - subsequent calls only see new logs
            - For network errors, check the 'source' field (e.g., "network")

        Example workflow:
            navigate("https://example.com")
            get_console_logs()  # Check for load-time errors
            click(selector)
            get_console_logs()  # Check for interaction errors
        """
        driver = await self.get_browser()

        logger.info('Getting browser console logs')

        # Get browser logs (clears buffer after retrieval)
        try:
            raw_logs = await asyncio.to_thread(driver.get_log, 'browser')
        except WebDriverException:
            logger.exception('Failed to get console logs')
            return ConsoleLogsResult(
                logs=[],
                total_count=0,
            )

        # Parse raw logs into structured entries
        entries: list[ConsoleLogEntry] = []
        level_counts = {'SEVERE': 0, 'WARNING': 0, 'INFO': 0}

        # Level hierarchy for filtering
        level_hierarchy = {'SEVERE': 3, 'WARNING': 2, 'INFO': 1}
        min_level = level_hierarchy.get(level_filter, 0) if level_filter and level_filter != 'ALL' else 0

        # Compile pattern if provided
        pattern_regex = re.compile(pattern, re.IGNORECASE) if pattern else None

        for log in raw_logs:
            level = log.get('level', 'INFO')
            message = log.get('message', '')
            source = log.get('source', '')
            timestamp = log.get('timestamp', 0)

            # Normalize level (Chrome sometimes uses lowercase)
            level = level.upper()
            if level not in level_counts:
                level = 'INFO'  # Default unknown levels to INFO

            # Count all logs by level (before filtering)
            level_counts[level] += 1

            # Apply level filter
            if min_level > 0 and level_hierarchy.get(level, 0) < min_level:
                continue

            # Apply pattern filter
            if pattern_regex and not pattern_regex.search(message):
                continue

            entries.append(
                ConsoleLogEntry(
                    level=level,
                    message=message,
                    source=source,
                    timestamp=timestamp,
                ),
            )

        result = ConsoleLogsResult(
            logs=entries,
            total_count=len(raw_logs),
            severe_count=level_counts['SEVERE'],
            warning_count=level_counts['WARNING'],
            info_count=level_counts['INFO'],
        )

        # Log summary
        if result.total_count > 0:
            logger.info(
                'Console logs: %s errors, %s warnings, %s info (%s returned after filtering)',
                result.severe_count,
                result.warning_count,
                result.info_count,
                len(entries),
            )
        else:
            logger.info('No console logs captured')

        return result

    @tool_registry.register_tool
    async def clear_proxy(
        self,
    ) -> ClearProxyResult:
        """Clear proxy configuration and return to direct connection.

        Stops the mitmproxy subprocess and clears proxy settings.

        Returns:
            Status dict confirming proxy cleared
        """
        logger.info('Clearing proxy configuration')

        # Close browser first
        await self.close_browser()

        # Stop mitmproxy process
        if self.state.mitmproxy_process:
            self.state.mitmproxy_process.terminate()
            try:
                self.state.mitmproxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.state.mitmproxy_process.kill()
            self.state.mitmproxy_process = None
            logger.info('mitmproxy stopped')

        self.state.proxy_config = None

        return ClearProxyResult(status='proxy_cleared')

    @tool_registry.register_tool
    async def set_blocked_urls(self, urls: Sequence[str]) -> SetBlockedURLsResult:
        """Block network requests matching URL patterns via CDP.

        Uses Chrome DevTools Protocol Network.setBlockedURLs to block requests
        at the browser's network layer before they hit the network.

        Each call REPLACES the blocked URL list. Pass an empty list to clear.
        Blocks persist across navigations but reset on fresh_browser=True.

        Args:
            urls: URL patterns to block. Wildcards supported:
                - '*cdn.segment.com*' blocks all Segment analytics
                - '*.png' blocks all PNG images
                - [] clears all blocks

        Returns:
            SetBlockedURLsResult with count and echoed patterns

        Example:
            set_blocked_urls(['*cdn.segment.com*', '*google-analytics.com*'])
            navigate('https://example.com')  # Loads without analytics
            set_blocked_urls([])  # Clear all blocks
        """
        driver = await self.get_browser()

        await asyncio.to_thread(driver.execute_cdp_cmd, 'Network.enable', {})
        await asyncio.to_thread(driver.execute_cdp_cmd, 'Network.setBlockedURLs', {'urls': list(urls)})

        if urls:
            logger.info('Blocked %s URL pattern(s): %s', len(urls), urls)
        else:
            logger.info('Cleared all URL blocks')

        return SetBlockedURLsResult(
            blocked_url_count=len(urls),
            patterns=list(urls),
        )

    @tool_registry.register_tool
    async def execute_javascript(self, code: str, timeout_ms: int = 30000) -> JavaScriptResult:
        """Execute JavaScript in the browser and return the result.

        Evaluates a JavaScript expression in the current page context.
        For multiple statements, wrap in an IIFE (Immediately Invoked Function Expression).

        Args:
            code: JavaScript expression to evaluate.
                  For multiple statements: (() => { const x = 1; return x; })()
            timeout_ms: Maximum execution time in milliseconds. 0 disables timeout. Default: 30000.

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

        Soft Navigation Pattern (preserve JS state across URL changes):
            For single-page apps or when you need to preserve JavaScript state
            (interceptors, global variables) across URL changes, use history.pushState:

            execute_javascript('''
                history.pushState({}, '', '/new-path');
                window.dispatchEvent(new PopStateEvent('popstate'));
            ''')

            This changes the URL without page reload, preserving all JS context.
            Only works for same-origin URLs. For cross-origin, use navigate()
            with init_scripts for persistent instrumentation.

        Notes:
            - Promises are automatically awaited
            - DOM nodes and functions cannot be serialized (return null with explanation)
            - BigInt and Symbol are converted to strings
            - Map → object, Set → array, circular references → "[Circular Reference]"
            - Sites with strict CSP may block execution
        """
        driver = await self.get_browser()

        logger.info('Executing JavaScript (%s chars)', len(code))

        # Escape user code as JSON string to prevent injection
        # json.dumps handles quotes, backslashes, newlines, etc.
        escaped_code = json.dumps(code)

        # Build the async wrapper for Selenium's execute_async_script
        # (uses safe_serialize.js loaded from scripts/)
        async_script = build_execute_javascript_async_script(escaped_code)

        try:
            # Use asyncio.wait_for for Python-side timeout
            # timeout_ms=0 means no timeout (wait indefinitely)
            if timeout_ms > 0:
                result = await asyncio.wait_for(
                    asyncio.to_thread(driver.execute_async_script, async_script),
                    timeout=timeout_ms / 1000,
                )
            else:
                result = await asyncio.to_thread(driver.execute_async_script, async_script)

            # Log result type for debugging
            result_type = result.get('result_type', 'unknown') if isinstance(result, dict) else 'unknown'
            success = result.get('success', False) if isinstance(result, dict) else False

            if success:
                logger.info('JS execution successful: %s', result_type)
            else:
                error = result.get('error', 'Unknown error') if isinstance(result, dict) else 'Unknown error'
                logger.warning('JS execution failed: %s', error)

            return JavaScriptResult(**result)

        except TimeoutError:
            logger.warning(
                'JS execution timed out after %sms',
                timeout_ms,
            )  # exception_safety_linter.py: logger-no-exc-info — TimeoutError has no useful traceback
            return JavaScriptResult(
                success=False,
                result_type='unserializable',
                error=f'Execution exceeded {timeout_ms}ms timeout',
                error_type='timeout',
            )
        except WebDriverException as e:
            logger.exception('JS execution WebDriver error')
            return JavaScriptResult(
                success=False,
                result_type='unserializable',
                error=str(e),
                error_type='execution',
            )
        except Exception as e:  # exception_safety_linter.py: swallowed-exception — error captured in JavaScriptResult
            logger.warning('JS execution unexpected error: %s', e, exc_info=True)
            return JavaScriptResult(
                success=False,
                result_type='unserializable',
                error=str(e),
                error_type='execution',
            )

    @tool_registry.register_tool
    async def export_chrome_profile_state(
        self,
        output_file: str,
        chrome_profile: str = 'Default',
        include_session_storage: bool = True,
        include_indexeddb: bool = False,
        origins_filter: Sequence[str] | None = None,
        live_session_storage_via_applescript: bool = False,
    ) -> ChromeProfileStateExportResult:
        """Export profile state from Chrome's profile files for use in automation.

        Captures cookies, localStorage, sessionStorage, and optionally IndexedDB
        from a standalone Chrome browser's profile files. Works with running Chrome.
        Outputs Playwright-compatible JSON for use with profile_state_file.

        This complements save_profile_state() which exports from a Selenium-controlled
        browser. Use this when you've logged in manually in Chrome and want to
        capture that authenticated profile state for automation.

        Workflow:
            1. Log into websites in normal Chrome browser (handles CAPTCHA, MFA)
            2. export_chrome_profile_state("auth.json")  # Capture profile state
            3. navigate_with_profile_state(url, profile_state_file="auth.json")  # Restore

        Args:
            output_file: Path to save JSON file (e.g., "auth.json")
            chrome_profile: Chrome profile name ("Default", "Profile 1", etc.)
            include_session_storage: Include sessionStorage (default True)
            include_indexeddb: Include IndexedDB records (default False, can be 200MB+)
            origins_filter: Only export origins matching these patterns
                           (e.g., ["github.com", "google.com"])
            live_session_storage_via_applescript: If True, extract live sessionStorage
                from running Chrome tabs via AppleScript. Defaults to False because
                Chrome suspends background tabs, causing AppleScript to hang on
                inactive tabs. WARNING: AppleScript extracts from ALL Chrome windows
                regardless of profile. If multiple profiles are open, sessionStorage
                may include data from other profiles. Requires one-time Chrome setting:
                View > Developer > Allow JavaScript from Apple Events.

        Returns:
            ChromeProfileStateExportResult with counts, session_storage_source, and warnings

        Storage Types (matches save_profile_state):
            - Cookies: Full attributes including sameSite
            - localStorage: All origins
            - sessionStorage: Live from Chrome tabs (default) or from disk files

        Limitations:
            - macOS only (Windows/Linux untested)
            - First run prompts for Keychain access - click "Always Allow"
            - Live sessionStorage requires one-time Chrome setting:
              View > Developer > Allow JavaScript from Apple Events

        Security:
            Output file created with 0o600 permissions (owner read/write only).
            Contains sensitive auth tokens - treat as credentials.
        """

        logger.info("Exporting Chrome profile state from profile '%s' to %s", chrome_profile, output_file)

        # Convert Sequence to list for the export function
        filter_list = list(origins_filter) if origins_filter else None

        # Call the sync export function in a thread
        result = await asyncio.to_thread(
            chrome_profile_state_export.export_chrome_profile_state,
            output_file=output_file,
            chrome_profile=chrome_profile,
            include_session_storage=include_session_storage,
            include_indexeddb=include_indexeddb,
            origins_filter=filter_list,
            live_session_storage_via_applescript=live_session_storage_via_applescript,
        )

        # Log summary
        parts = [f'{result.cookie_count} cookies']
        if result.local_storage_keys > 0:
            parts.append(f'{result.local_storage_keys} localStorage keys')
        if result.session_storage_keys > 0:
            source_label = 'live' if result.session_storage_source == 'live' else 'disk'
            parts.append(f'{result.session_storage_keys} sessionStorage keys ({source_label})')
        if result.indexeddb_origins > 0:
            parts.append(f'{result.indexeddb_origins} IndexedDB origins')

        logger.info('Exported %s across %s origins to %s', ', '.join(parts), result.origin_count, result.path)

        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)

        return ChromeProfileStateExportResult(
            path=result.path,
            cookie_count=result.cookie_count,
            origin_count=result.origin_count,
            local_storage_keys=result.local_storage_keys,
            session_storage_keys=result.session_storage_keys,
            indexeddb_origins=result.indexeddb_origins,
            session_storage_source=result.session_storage_source,
            warnings=list(result.warnings),
        )

    @tool_registry.register_tool
    async def navigate(
        self,
        url: str,
        fresh_browser: bool = False,
        enable_har_capture: bool = False,
        init_scripts: Sequence[str] | None = None,
        browser: Browser | None = None,
    ) -> NavigationResult:
        """Load a URL and establish browser session. Entry point for all browser automation.

        After navigation completes, call get_aria_snapshot('body') to understand page structure
        before interacting with elements.

        Args:
            url: Full URL (http:// or https://)
            fresh_browser: If True, creates clean session (no cache/cookies)
            enable_har_capture: If True, enables performance logging for HAR export.
                               Requires fresh_browser=True (adds overhead, must be set at browser init).
            init_scripts: JavaScript code to run before every page load (requires fresh_browser=True).
                         Scripts persist for all navigations until next fresh_browser=True.
                         Use for API interceptors, environment patching.
            browser: Which browser to use - "chrome" or "chromium". Defaults to the currently
                    running browser, or "chromium" if none is running.
                    Use "chromium" to avoid AppleScript targeting conflicts when
                    your personal Chrome is running (different bundle ID).

        Example - API interception:
            navigate(
                "https://example.com",
                fresh_browser=True,
                init_scripts=['''
                    window.__apiCapture = [];
                    const originalFetch = window.fetch;
                    window.fetch = async (...args) => {
                        window.__apiCapture.push({url: args[0], time: Date.now()});
                        return originalFetch(...args);
                    };
                ''']
            )
            # Navigate around - interceptor persists
            navigate("https://example.com/account")
            # Retrieve captured APIs via execute_javascript('window.__apiCapture')

        Note:
            init_scripts run BEFORE page scripts in every frame (including iframes).
            Do not modify navigator.webdriver, navigator.languages, navigator.plugins,
            or window.chrome - these are reserved for bot detection evasion.

        Returns:
            NavigationResult with current_url and title

        Next steps:
            1. get_aria_snapshot('body') - understand page structure
            2. get_interactive_elements() - find specific elements to click
            3. click(selector) - interact with elements

        For performance investigation:
            Use get_resource_timings() after navigation to measure page load timing.
            Use export_har() for detailed HTTP transaction data (requires enable_har_capture=True).

        Note: Single-page apps may ignore URL parameters. Page elements control navigation state.

        Blob URLs: blob:https://... URLs are supported but only work if the blob was created
        in the current page context. Blob URLs are ephemeral in-memory resources (PDFs, images,
        file downloads) and cannot be accessed from a different browsing context.
        """
        # Resolve browser: use current if running, otherwise default to 'chromium'
        if browser is None:
            browser = self.state.current_browser or 'chromium'

        if not url.startswith(VALID_URL_PREFIXES):
            raise fastmcp.exceptions.ValidationError(
                'URL must start with http://, https://, file://, about:, data:, or blob:',
            )

        if self.state.current_browser is not None and browser != self.state.current_browser and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                f"Browser changed from '{self.state.current_browser}' to '{browser}'. "
                'Pass fresh_browser=True to restart with the new browser.',
            )

        if enable_har_capture and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                'enable_har_capture requires fresh_browser=True (performance logging must be set at browser init)',
            )

        if init_scripts and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                'init_scripts requires fresh_browser=True (scripts must be registered before first navigation)',
            )

        logger.info(
            'Navigating to %s%s%s%s',
            url,
            ' (fresh browser)' if fresh_browser else '',
            ' (HAR capture enabled)' if enable_har_capture else '',
            f' ({len(init_scripts)} init scripts)' if init_scripts else '',
        )

        if fresh_browser:
            await self.close_browser()

        driver = await self.get_browser(enable_har_capture=enable_har_capture, browser=browser)

        # Install user init scripts (after browser creation, before navigation)
        # Scripts registered here run on EVERY new document in this session
        if init_scripts:
            for script in init_scripts:
                await asyncio.to_thread(
                    driver.execute_cdp_cmd,
                    'Page.addScriptToEvaluateOnNewDocument',
                    {'source': script},
                )

        # Install response body capture interceptor for HAR export
        # Must run BEFORE first navigation to capture all fetch/XHR responses
        await _install_response_body_capture_if_needed(driver, self, enable_har_capture, 'navigate')

        # PRE-ACTION: Capture localStorage before navigating away
        # (CDP can't query departed origins - frame is gone after navigation)
        await _capture_current_origin_storage(self, driver)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        # Track the final origin after redirects for multi-origin storage capture
        final_url = driver.current_url
        self.state.origin_tracker.add_origin(final_url)

        logger.info('Successfully navigated to %s (tracked origins: %s)', final_url, len(self.state.origin_tracker))

        # Lazy restore: if navigate_with_profile_state() was called previously, restore
        # storage for current origin. This handles multi-origin sessions where the user
        # navigates to different origins after the initial session import.
        # The helper is idempotent and checks restored_origins to avoid double-restore.
        await _restore_pending_profile_state_for_current_origin(self, driver)

        return NavigationResult(current_url=driver.current_url, title=driver.title)

    @tool_registry.register_tool
    async def navigate_with_profile_state(
        self,
        url: str,
        # Profile state source (one required)
        profile_state_file: str | None = None,
        chrome_profile: str | None = None,
        origins_filter: Sequence[str] | None = None,
        live_session_storage_via_applescript: bool = False,
        # Browser configuration (all fresh_browser capabilities)
        browser: Browser | None = None,
        enable_har_capture: bool = False,
        init_scripts: Sequence[str] | None = None,
    ) -> NavigationResult:
        """Launch fresh browser with imported profile state and navigate.

        PERMISSION SCOPE: This tool imports sensitive authentication data.
        Requires separate approval from navigate().

        Use this for authenticated automation where you need to:
        - Import a previously saved profile state (from save_profile_state or export_chrome_profile_state)
        - Import profile state directly from a Chrome profile

        Profile State Sources (exactly one required):
        - profile_state_file: Load from ProfileState JSON file
        - chrome_profile: Export and load from Chrome profile

        Args:
            url: URL to navigate to after profile state import
            profile_state_file: Path to ProfileState JSON file (from save_profile_state or export_chrome_profile_state)
            chrome_profile: Chrome profile name to import from ("Default", "Profile 1", etc.)
            origins_filter: Only import origins matching these patterns (e.g., ["amazon.com"])
                           Works with both profile_state_file and chrome_profile.
            live_session_storage_via_applescript: If True, extract live sessionStorage from running
                Chrome tabs via AppleScript when using chrome_profile. Default False.
                WARNING: AppleScript extracts from ALL Chrome windows regardless of profile.
                If multiple profiles are open, sessionStorage may include data from other profiles.
                Requires Chrome setting: View > Developer > Allow JavaScript from Apple Events.
                Only used with chrome_profile, ignored with profile_state_file.
            browser: Which browser to use - "chrome" or "chromium". Defaults to the currently
                running browser, or "chromium" if none is running.
                Use "chromium" to avoid AppleScript targeting conflicts when
                your personal Chrome is running (different bundle ID).
            enable_har_capture: Enable performance logging for HAR export
            init_scripts: JavaScript to inject before every page load

        Returns:
            NavigationResult with current_url and title

        Example - From file:
            navigate_with_profile_state("https://github.com", profile_state_file="auth.json")

        Example - From Chrome profile:
            navigate_with_profile_state("https://amazon.com",
                                        chrome_profile="Profile 1",
                                        origins_filter=["amazon.com"])

        Note:
            This tool always starts a fresh browser (implicit fresh_browser=True).

        Blob URLs: blob:https://... URLs are supported but only work if the blob was created
        in the current page context. Blob URLs are ephemeral in-memory resources (PDFs, images,
        file downloads) and cannot be accessed from a different browsing context.
        """
        timer = Timer()

        # Validate URL
        if not url.startswith(VALID_URL_PREFIXES):
            raise fastmcp.exceptions.ValidationError(
                'URL must start with http://, https://, file://, about:, data:, or blob:',
            )

        # Validate profile state source - exactly one required
        has_file = profile_state_file is not None
        has_chrome = chrome_profile is not None

        if not has_file and not has_chrome:
            raise fastmcp.exceptions.ValidationError(
                'Exactly one of profile_state_file or chrome_profile is required. '
                'Provide a profile state source to import.',
            )

        if has_file and has_chrome:
            raise fastmcp.exceptions.ValidationError(
                'Cannot specify both profile_state_file and chrome_profile. Choose one profile state source.',
            )

        # Log what we're doing
        if has_chrome:
            logger.info(
                "Importing profile state from Chrome profile '%s' and navigating to %s%s%s%s",
                chrome_profile,
                url,
                f' (filtering: {origins_filter})' if origins_filter else '',
                ' (HAR capture enabled)' if enable_har_capture else '',
                f' ({len(init_scripts)} init scripts)' if init_scripts else '',
            )
        else:
            logger.info(
                'Loading profile state from %s and navigating to %s%s%s%s',
                profile_state_file,
                url,
                f' (filtering: {origins_filter})' if origins_filter else '',
                ' (HAR capture enabled)' if enable_har_capture else '',
                f' ({len(init_scripts)} init scripts)' if init_scripts else '',
            )

        # Load profile state from the appropriate source
        profile_state: ProfileState

        if has_chrome and chrome_profile:
            # Export from Chrome profile to temp file, then load
            # Use TemporaryDirectory as context manager for automatic cleanup
            with tempfile.TemporaryDirectory(prefix='chrome_profile_state_') as temp_dir:
                temp_file_path = Path(temp_dir) / 'profile_state.json'
                filter_list = list(origins_filter) if origins_filter else None

                # Use Python-level stdout redirect to capture ccl_chromium_reader's
                # "Error decoding..." print statements without affecting MCP's fd-level I/O
                def _export_with_stdout_captured() -> None:
                    with contextlib.redirect_stdout(io.StringIO()):
                        chrome_profile_state_export.export_chrome_profile_state(
                            output_file=str(temp_file_path),
                            chrome_profile=chrome_profile,
                            include_session_storage=True,
                            include_indexeddb=False,  # IndexedDB schema issues
                            origins_filter=filter_list,
                            live_session_storage_via_applescript=live_session_storage_via_applescript,
                        )

                await asyncio.to_thread(_export_with_stdout_captured)
                profile_state = await _load_profile_state_from_file(str(temp_file_path))

            logger.info("Exported %s cookies from Chrome profile '%s'", len(profile_state.cookies), chrome_profile)

        elif profile_state_file:
            # Load from file directly
            profile_state = await _load_profile_state_from_file(profile_state_file)

            logger.info('Loaded %s cookies from %s', len(profile_state.cookies), profile_state_file)

        else:
            # This should not happen due to earlier validation
            raise fastmcp.exceptions.ValidationError('No valid profile state source specified.')

        # Apply origins_filter to loaded profile state if specified
        if origins_filter and has_file:
            # Filter cookies
            filtered_cookies = []
            for cookie in profile_state.cookies:
                cookie_domain = cookie.domain.lower().strip('.')
                for pattern in origins_filter:
                    pattern_clean = pattern.lower().strip('.')
                    if cookie_domain == pattern_clean or cookie_domain.endswith('.' + pattern_clean):
                        filtered_cookies.append(cookie)
                        break

            # Filter origins (dict-based in new format)
            filtered_origins: dict[str, ProfileStateOriginStorage] = {}
            for origin_url, origin_data in profile_state.origins.items():
                origin_domain = origin_url.lower()
                # Extract domain from origin URL
                if '://' in origin_domain:
                    origin_domain = origin_domain.split('://', 1)[1]
                    origin_domain = origin_domain.split('/', 1)[0]
                    origin_domain = origin_domain.split(':', 1)[0]

                for pattern in origins_filter:
                    pattern_clean = pattern.lower().strip('.')
                    if origin_domain == pattern_clean or origin_domain.endswith('.' + pattern_clean):
                        filtered_origins[origin_url] = origin_data
                        break

            # Create filtered profile state
            profile_state = ProfileState(
                cookies=list(filtered_cookies),
                origins=filtered_origins,
            )

            logger.info(
                'Filtered to %s cookies and %s origins matching %s',
                len(filtered_cookies),
                len(filtered_origins),
                origins_filter,
            )

        # Resolve browser before close_browser() clears current_browser
        if browser is None:
            browser = self.state.current_browser or 'chromium'

        # Always start fresh browser for session import
        await self.close_browser()

        # Get browser with configuration
        driver = await self.get_browser(enable_har_capture=enable_har_capture, browser=browser)

        # Build and register storage init script for localStorage/sessionStorage
        # This runs BEFORE page JavaScript on every new document (Playwright-style)
        storage_init_script = build_storage_init_script(profile_state)
        if storage_init_script:
            await asyncio.to_thread(
                driver.execute_cdp_cmd,
                'Page.addScriptToEvaluateOnNewDocument',
                {'source': storage_init_script},
            )
            # Count storage entries for logging
            storage_entry_count = sum(
                len(origin_data.local_storage or {}) + len(origin_data.session_storage or {})
                for origin_data in profile_state.origins.values()
            )
            logger.info(
                'Registered storage init script (%s entries across %s origins)',
                storage_entry_count,
                len(profile_state.origins),
            )

        # Install user init scripts (after storage script, before navigation)
        if init_scripts:
            for script in init_scripts:
                await asyncio.to_thread(
                    driver.execute_cdp_cmd,
                    'Page.addScriptToEvaluateOnNewDocument',
                    {'source': script},
                )

        # Install response body capture interceptor for HAR export
        # Must run BEFORE first navigation to capture all fetch/XHR responses
        await _install_response_body_capture_if_needed(
            driver,
            self,
            enable_har_capture,
            'navigate_with_profile_state',
        )

        # Inject cookies via CDP BEFORE navigation
        cookies_injected = await _inject_cookies_via_cdp(driver, profile_state.cookies)

        logger.info('Injected %s cookies via CDP', cookies_injected)

        # PRE-ACTION: Capture localStorage before navigating away
        await _capture_current_origin_storage(self, driver)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        # Track the final origin after redirects
        final_url = driver.current_url
        self.state.origin_tracker.add_origin(final_url)

        logger.info('Successfully navigated to %s (tracked origins: %s)', final_url, len(self.state.origin_tracker))

        # Setup lazy restore for localStorage/sessionStorage/IndexedDB
        await _setup_pending_profile_state(self, profile_state)

        # Restore storage for current origin immediately
        await _restore_pending_profile_state_for_current_origin(self, driver)

        return NavigationResult(
            current_url=driver.current_url,
            title=driver.title,
            elapsed_seconds=round(timer.elapsed(), 3),
        )

    @tool_registry.register_tool
    async def export_har(
        self,
        filename: str,
        include_response_bodies: bool = False,
        max_body_size_mb: int = 10,
    ) -> HARExportResult:
        driver = await self.get_browser()

        logger.info('Exporting HAR to %s', filename)
        errors: list[str] = []

        # Get performance logs (clears buffer - subsequent calls return only newer entries)
        try:
            logs = await asyncio.to_thread(driver.get_log, 'performance')
        except WebDriverException as e:
            errors.append(f'Failed to get performance logs: {e}')
            logs = []

        if not logs:
            logger.info('No performance logs available')
            har_path = self.state.capture_dir / filename
            empty_har = {
                'log': {
                    'version': '1.2',
                    'creator': {'name': 'selenium-browser-automation', 'version': '1.0'},
                    'entries': [],
                },
            }
            har_path.write_text(json.dumps(empty_har, indent=2))
            return HARExportResult(
                path=str(har_path),
                entry_count=0,
                size_bytes=len(json.dumps(empty_har)),
                has_errors=True,
                errors=['No performance logs available. Navigate to pages first.'],
            )

        # Parse and filter Network.* events
        transactions: dict[str, dict[str, Any]] = {}
        for entry in logs:
            try:
                log_data = json.loads(entry['message'])
                message = log_data.get('message', {})
                method = message.get('method', '')
                params = message.get('params', {})

                if not method.startswith('Network.'):
                    continue

                request_id = params.get('requestId')
                if not request_id:
                    continue

                if method == 'Network.requestWillBeSent':
                    req = params.get('request', {})
                    transactions[request_id] = {
                        'request': req,
                        'wall_time': params.get('wallTime'),
                        'timestamp': params.get('timestamp'),
                        'initiator': params.get('initiator'),
                    }

                elif method == 'Network.responseReceived':
                    if request_id in transactions:
                        resp = params.get('response', {})
                        transactions[request_id]['response'] = resp
                        transactions[request_id]['resource_type'] = params.get('type')

                elif method == 'Network.loadingFinished':
                    if request_id in transactions:
                        transactions[request_id]['encoded_data_length'] = params.get('encodedDataLength', 0)
                        transactions[request_id]['complete'] = True

            except (json.JSONDecodeError, KeyError):
                continue

        # Convert transactions to HAR entries
        har_entries = []

        for request_id, txn in transactions.items():
            if 'response' not in txn:
                continue  # Skip incomplete transactions

            req = txn.get('request', {})
            resp = txn.get('response', {})
            timing = resp.get('timing', {})

            # Convert wallTime (seconds since epoch) to ISO 8601
            wall_time = txn.get('wall_time')
            if wall_time:
                dt = datetime.fromtimestamp(wall_time, tz=UTC)
                started_datetime = dt.isoformat().replace('+00:00', 'Z')
            else:
                started_datetime = datetime.now(UTC).isoformat().replace('+00:00', 'Z')

            # Convert headers dict to HAR array format
            def headers_to_har(headers_dict: JsonObject | None) -> Sequence[Mapping[str, str]]:
                if not headers_dict:
                    return []
                return [{'name': k, 'value': str(v)} for k, v in headers_dict.items()]

            # Parse query string from URL
            def parse_query_string(url: str) -> Sequence[Mapping[str, str]]:
                parsed = urlparse(url)
                query_params = parse_qs(parsed.query)
                return [{'name': name, 'value': value} for name, values in query_params.items() for value in values]

            # Convert CDP timing to HAR timing (duration in ms)
            def convert_timing(timing_obj: JsonObject) -> Mapping[str, float | int]:
                def safe_duration(start_key: str, end_key: str) -> float | int:
                    start = timing_obj.get(start_key)
                    end = timing_obj.get(end_key)
                    if start is not None and end is not None and start >= 0 and end >= 0:
                        duration: float = max(0, end - start)
                        return duration
                    return -1

                return {
                    'blocked': -1,  # Not directly available
                    'dns': safe_duration('dnsStart', 'dnsEnd'),
                    'connect': safe_duration('connectStart', 'connectEnd'),
                    'ssl': safe_duration('sslStart', 'sslEnd'),
                    'send': safe_duration('sendStart', 'sendEnd'),
                    'wait': safe_duration('sendEnd', 'receiveHeadersEnd'),
                    'receive': 0,  # Would need loadingFinished timing
                }

            har_timing = convert_timing(timing)

            # Calculate total time
            total_time = sum(v for v in har_timing.values() if v >= 0)

            har_entry = {
                'startedDateTime': started_datetime,
                'time': total_time,
                'request': {
                    'method': req.get('method', 'GET'),
                    'url': req.get('url', ''),
                    'httpVersion': resp.get('protocol', 'HTTP/1.1'),
                    'headers': headers_to_har(req.get('headers', {})),
                    'queryString': parse_query_string(req.get('url', '')),
                    'cookies': [],  # Would need to parse from headers
                    'headersSize': -1,
                    'bodySize': len(req.get('postData', '')) if req.get('postData') else 0,
                },
                'response': {
                    'status': resp.get('status', 0),
                    'statusText': resp.get('statusText', ''),
                    'httpVersion': resp.get('protocol', 'HTTP/1.1'),
                    'headers': headers_to_har(resp.get('headers', {})),
                    'cookies': [],
                    'content': {
                        'size': resp.get('encodedDataLength', 0),
                        'mimeType': resp.get('mimeType', ''),
                    },
                    'redirectURL': '',
                    'headersSize': -1,
                    'bodySize': txn.get('encoded_data_length', -1),
                },
                'cache': {},
                'timings': har_timing,
                'serverIPAddress': resp.get('remoteIPAddress', ''),
                'connection': resp.get('connectionId', ''),
            }

            # Add POST data if present
            if req.get('postData'):
                har_entry['request']['postData'] = {
                    'mimeType': req.get('headers', {}).get('Content-Type', ''),
                    'text': req.get('postData'),
                }

            # Fetch response body if requested (configurable size limit)
            if include_response_bodies:
                mime_type = resp.get('mimeType', '')
                body_size = txn.get('encoded_data_length', 0)
                max_body_bytes = max(1, min(max_body_size_mb, 50)) * 1024 * 1024  # 1-50MB
                should_fetch = body_size < max_body_bytes and (
                    'json' in mime_type or 'text' in mime_type or 'xml' in mime_type or 'javascript' in mime_type
                )
                if should_fetch:
                    body_source: str | None = None

                    # First, try CDP getResponseBody (works for static resources)
                    try:
                        body_result = await asyncio.to_thread(
                            driver.execute_cdp_cmd,
                            'Network.getResponseBody',
                            {'requestId': request_id},
                        )
                        if body_result.get('body'):
                            har_entry['response']['content']['text'] = body_result['body']
                            if body_result.get('base64Encoded'):
                                har_entry['response']['content']['encoding'] = 'base64'
                            body_source = 'cdp'
                    except WebDriverException:
                        # CDP failed - Chrome likely GC'd the response body
                        # Will try JavaScript interceptor fallback below
                        pass

                    # Fallback: Check JavaScript interceptor capture
                    # No try/except here - let programming errors bubble up
                    if body_source is None and self.state.response_body_capture_enabled:
                        intercepted = await _lookup_intercepted_body(
                            driver,
                            url=req.get('url', ''),
                            method=req.get('method', 'GET'),
                            cdp_timestamp=txn.get('wall_time'),
                        )
                        if intercepted is not None and intercepted.get('body'):
                            har_entry['response']['content']['text'] = intercepted['body']
                            if intercepted.get('base64Encoded'):
                                har_entry['response']['content']['encoding'] = 'base64'
                            body_source = 'interceptor'
                            if intercepted.get('truncated'):
                                errors.append(f'Response truncated at 10MB: {req.get("url", "")[:80]}')

                    # Report if body unavailable from both sources
                    if body_source is None:
                        method = req.get('method', 'GET')
                        status = resp.get('status', 0)
                        url_preview = req.get('url', '')[:100]

                        if method == 'OPTIONS':
                            # CORS preflight - no body expected
                            errors.append(f'[{method} {status}] CORS preflight: {url_preview}')
                        elif status == 204:
                            # HTTP 204 No Content - no body by definition
                            errors.append(f'[{method} {status}] No Content: {url_preview}')
                        else:
                            # Unexpected - we wanted a body but couldn't get it
                            errors.append(f'[{method} {status}] body unavailable: {url_preview}')

            har_entries.append(har_entry)

        # Build HAR structure
        har = {
            'log': {
                'version': '1.2',
                'creator': {'name': 'selenium-browser-automation', 'version': '1.0'},
                'entries': har_entries,
            },
        }

        # Save to file
        har_path = self.state.capture_dir / filename
        har_json = json.dumps(har, indent=2)
        har_path.write_text(har_json)

        logger.info('Exported %s entries to %s', len(har_entries), har_path)

        return HARExportResult(
            path=str(har_path),
            entry_count=len(har_entries),
            size_bytes=len(har_json),
            has_errors=len(errors) > 0,
            errors=errors,
        )

    @tool_registry.register_tool
    async def configure_proxy(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
    ) -> ConfigureProxyResult:
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
        logger.info('Configuring proxy via mitmproxy: %s:%s', host, port)

        # Close existing browser and mitmproxy
        await self.close_browser()

        # Stop any existing mitmproxy process
        if self.state.mitmproxy_process:
            self.state.mitmproxy_process.terminate()
            try:
                self.state.mitmproxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.state.mitmproxy_process.kill()
            self.state.mitmproxy_process = None

        # Store proxy config
        self.state.proxy_config = {
            'host': host,
            'port': str(port),
            'username': username,
            'password': password,
        }

        # Start mitmproxy with upstream authentication
        # mitmproxy handles auth with Bright Data, Chrome connects to localhost:8080
        upstream_url = f'http://{host}:{port}'
        upstream_auth = f'{username}:{password}'

        try:
            self.state.mitmproxy_process = subprocess.Popen(
                [
                    'mitmdump',
                    '--mode',
                    f'upstream:{upstream_url}',
                    '--upstream-auth',
                    upstream_auth,
                    '--listen-host',
                    '127.0.0.1',
                    '--listen-port',
                    '8080',
                    '--ssl-insecure',  # Accept upstream proxy's certs
                    '--quiet',  # Reduce log noise
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Give mitmproxy time to start
            await asyncio.sleep(2)

            # Check if process is still running
            if self.state.mitmproxy_process.poll() is not None:
                stderr = (
                    self.state.mitmproxy_process.stderr.read().decode() if self.state.mitmproxy_process.stderr else ''
                )
                raise RuntimeError(f'mitmproxy failed to start: {stderr}')

            logger.info('mitmproxy started on localhost:8080')

        except FileNotFoundError:
            self.state.proxy_config = None
            raise fastmcp.exceptions.ToolError('mitmproxy not found. Install with: pip install mitmproxy') from None
        except Exception as e:
            self.state.proxy_config = None
            if self.state.mitmproxy_process:
                self.state.mitmproxy_process.kill()
                self.state.mitmproxy_process = None
            raise fastmcp.exceptions.ToolError(f'Failed to start mitmproxy: {e}') from e

        return ConfigureProxyResult(
            status='proxy_configured',
            host=host,
            port=port,
            note='Browser will use this proxy on next navigate()',
        )

    @tool_registry.register_tool
    async def save_profile_state(
        self,
        filename: str,
        include_indexeddb: bool = False,
    ) -> SaveProfileStateResult:
        """Export browser storage state to Playwright-compatible JSON for session persistence.

        Captures cookies, localStorage, and sessionStorage that maintain authenticated sessions.
        After logging in once, save storage state to reuse authentication in future sessions.

        Args:
            filename: Output filename (relative to cwd or absolute path).
                      Example: "marriott_auth.json"
            include_indexeddb: If True, capture IndexedDB databases for current origin.
                              Enable for apps using IndexedDB for auth (e.g., Firebase).
                              Default False for backward compatibility and performance.

        Returns:
            SaveProfileStateResult with path, cookie count, and metadata

        What's captured:
            - All cookies with full attributes (HttpOnly, Secure, SameSite, expires)
            - localStorage for all tracked origins (multi-origin via lazy capture)
            - sessionStorage for all tracked origins (multi-origin via lazy capture)
            - IndexedDB databases (if include_indexeddb=True, multi-origin via lazy capture)

        sessionStorage behavior:
            sessionStorage is session-scoped by browser design. Restored sessionStorage
            persists only for the lifetime of the browser context - closing the browser
            clears it. For cross-session persistence, use localStorage or cookies.

        Workflow:
            1. navigate("https://example.com/login", fresh_browser=True)
            2. [Complete login flow - click buttons, enter credentials, etc.]
            3. navigate("https://example.com/account")  # Navigate to authenticated page
            4. save_profile_state("example_auth.json", include_indexeddb=True)  # Export auth
            5. [Later, in new session:]
               navigate_with_profile_state("https://example.com/account",
                                           profile_state_file="example_auth.json")  # Restore auth

        Format:
            Saves in Playwright storageState JSON format for cross-tool compatibility.
            File can be used with Playwright, Puppeteer, or manually edited.

        Security:
            Storage state files contain authentication cookies and tokens.
            Treat them as credentials:
            - Never commit to version control
            - Encrypt at rest for long-term storage
            - Delete when no longer needed

        Limitations:
            - Tokens may expire between save and restore - re-authenticate if needed
        """

        driver = self.state.driver
        if driver is None:
            raise fastmcp.exceptions.ToolError('Browser not initialized. Call navigate() first to establish a session.')

        logger.info('Exporting storage state to %s', filename)

        # Get all cookies via CDP Network.getCookies
        cookies_result = await asyncio.to_thread(
            driver.execute_cdp_cmd,
            'Network.getCookies',
            {},  # Empty = get all cookies
        )

        cdp_cookies = cookies_result.get('cookies', [])

        # Convert CDP cookies to ProfileStateCookie models
        profile_cookies: list[ProfileStateCookie] = []
        for cookie in cdp_cookies:
            is_session = cookie.get('session', False)
            expires = -1.0 if is_session else cookie.get('expires', -1.0)

            same_site = cookie.get('sameSite', 'Lax')
            if same_site not in ('Strict', 'Lax', 'None'):
                same_site = 'Lax'

            profile_cookies.append(
                ProfileStateCookie(
                    name=cookie['name'],
                    value=cookie['value'],
                    domain=cookie['domain'],
                    path=cookie['path'],
                    expires=expires,
                    http_only=cookie.get('httpOnly', False),
                    secure=cookie.get('secure', False),
                    same_site=same_site,
                ),
            )

        # Get current origin (for result metadata)
        current_origin = await asyncio.to_thread(driver.execute_script, 'return window.location.origin')

        # =================================================================
        # Multi-Origin Storage Capture (localStorage + sessionStorage)
        # =================================================================
        # CDP DOMStorage.getDOMStorageItems requires an active frame for the origin.
        # For departed origins (navigated away), we use the cached data captured
        # during navigate(). For the current origin, we query CDP directly.
        # =================================================================

        tracked_origins = self.state.origin_tracker.get_origins()
        await _cdp_enable_domstorage(driver)

        # Build origins as dict[str, ProfileStateOriginStorage]
        origins_data: dict[str, ProfileStateOriginStorage] = {}
        total_localstorage_items = 0
        total_sessionstorage_items = 0
        indexeddb_databases_count = 0
        indexeddb_records_count = 0

        for origin in tracked_origins:
            if origin.startswith(('chrome://', 'about:', 'data:', 'blob:', 'file://')):
                continue

            # Capture localStorage: cache for departed origins, CDP for current
            if origin == current_origin:
                local_storage_items = await _cdp_get_storage(driver, origin, is_local=True)
            elif origin in self.state.local_storage_cache:
                local_storage_items = self.state.local_storage_cache[origin]
            else:
                local_storage_items = []

            # Capture sessionStorage: cache for departed origins, CDP for current
            if origin == current_origin:
                session_storage_items = await _cdp_get_storage(driver, origin, is_local=False)
            elif origin in self.state.session_storage_cache:
                session_storage_items = self.state.session_storage_cache[origin]
            else:
                session_storage_items = []

            # Capture IndexedDB if requested
            indexeddb_databases: list[dict[str, Any]] = []
            if include_indexeddb:
                if origin == current_origin:
                    async_wrapper = f"""
                        var callback = arguments[arguments.length - 1];
                        (function() {{ {INDEXEDDB_CAPTURE_SCRIPT} }})()
                            .then(function(r) {{ callback(r); }})
                            .catch(function(e) {{ callback([]); }});
                    """
                    indexeddb_result = await asyncio.to_thread(driver.execute_async_script, async_wrapper)
                    # execute_async_script returns Any; cast to expected JS structure
                    indexeddb_databases = cast(list[dict[str, Any]], indexeddb_result) if indexeddb_result else []
                elif origin in self.state.indexed_db_cache:
                    indexeddb_databases = self.state.indexed_db_cache[origin]

            # Only add origin if it has any storage
            if local_storage_items or session_storage_items or indexeddb_databases:
                # Convert array of {name, value} to dict[str, str]
                local_storage_dict = {item['name']: item['value'] for item in local_storage_items}
                session_storage_dict = (
                    {item['name']: item['value'] for item in session_storage_items} if session_storage_items else None
                )

                # Validate raw JS dicts through Pydantic model
                validated_indexeddb: list[ProfileStateIndexedDB] | None = (
                    [ProfileStateIndexedDB.model_validate(db) for db in indexeddb_databases]
                    if indexeddb_databases
                    else None
                )

                origins_data[origin] = ProfileStateOriginStorage(
                    local_storage=local_storage_dict,
                    session_storage=session_storage_dict,
                    indexed_db=validated_indexeddb,
                )

                total_localstorage_items += len(local_storage_items)
                total_sessionstorage_items += len(session_storage_items)

                # Count IndexedDB for metadata
                if indexeddb_databases:
                    for db in indexeddb_databases:
                        indexeddb_databases_count += 1
                        for store in db.get('objectStores', []):
                            indexeddb_records_count += len(store.get('records', []))

        # Build log message parts
        log_storage_parts = [
            f'{total_localstorage_items} localStorage',
            f'{total_sessionstorage_items} sessionStorage',
        ]
        if include_indexeddb:
            log_storage_parts.append(f'{indexeddb_databases_count} IndexedDB databases')

        logger.info(
            'Captured storage: %s across %s origins (of %s tracked)',
            ' + '.join(log_storage_parts),
            len(origins_data),
            len(tracked_origins),
        )

        # Build ProfileState with typed models
        profile_state = ProfileState(
            cookies=profile_cookies,
            origins=origins_data,
        )

        # Determine file path (allow absolute or relative to cwd)
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = Path.cwd() / filename

        # Save to file (by_alias=True for camelCase IndexedDB fields for JS compatibility)
        json_content = profile_state.model_dump_json(indent=2, by_alias=True)
        file_path.write_text(json_content)

        result = SaveProfileStateResult(
            path=str(file_path),
            cookies_count=len(profile_cookies),
            origins_count=len(origins_data),
            current_origin=current_origin,
            size_bytes=len(json_content),
            indexeddb_databases_count=indexeddb_databases_count if include_indexeddb else None,
            indexeddb_records_count=indexeddb_records_count if include_indexeddb else None,
            tracked_origins=tracked_origins,
        )

        # Build log message
        log_parts = [
            f'Saved {result.cookies_count} cookies',
            f'{total_localstorage_items} localStorage + {total_sessionstorage_items} sessionStorage items '
            f'across {len(origins_data)} origins',
        ]
        if include_indexeddb and indexeddb_databases_count > 0:
            log_parts.append(f'{indexeddb_databases_count} IndexedDB databases ({indexeddb_records_count} records)')

        logger.info('%s for %s to %s (%s bytes)', ' + '.join(log_parts), current_origin, file_path, result.size_bytes)

        return result


async def _lookup_intercepted_body(
    driver: webdriver.Chrome,
    url: str,
    method: str,
    cdp_timestamp: float | None,
) -> JsonObject | None:
    """Look up response body from JavaScript interceptor capture.

    Matches by URL + method + timestamp (within 5s window).
    CDP timestamps are seconds since epoch, JS timestamps are milliseconds.

    Args:
        driver: WebDriver instance
        url: Request URL to match
        method: HTTP method to match
        cdp_timestamp: CDP wallTime (seconds since epoch), or None

    Returns:
        Dict with 'body', 'base64Encoded', 'truncated' if found, else None
    """
    # Get captured responses from JavaScript
    captured = await asyncio.to_thread(
        driver.execute_script,
        'return window.__responseBodies || [];',
    )

    if not captured:
        return None

    # Convert CDP timestamp (seconds) to JS timestamp (milliseconds)
    cdp_timestamp_ms = (cdp_timestamp * 1000) if cdp_timestamp else None

    # Find best match by URL + method + closest timestamp
    best_match = None
    best_time_diff = float('inf')

    for entry in captured:
        # URL and method must match exactly
        if entry.get('url') != url:
            continue
        if entry.get('method', 'GET').upper() != method.upper():
            continue

        # Skip entries without response body (errors)
        if entry.get('responseBody') is None and entry.get('error'):
            continue

        # Timestamp matching (5 second window)
        entry_timestamp = entry.get('timestamp')  # milliseconds
        if cdp_timestamp_ms and entry_timestamp:
            time_diff = abs(entry_timestamp - cdp_timestamp_ms)
            if time_diff > 5000:  # More than 5 seconds apart
                continue
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_match = entry
        elif best_match is None:
            # No timestamp to compare, take first URL/method match
            best_match = entry

    if best_match is None:
        return None

    return {
        'body': best_match.get('responseBody'),
        'base64Encoded': best_match.get('base64Encoded', False),
        'truncated': best_match.get('truncated', False),
    }


async def _load_profile_state_from_file(file_path: str) -> ProfileState:
    """Load and validate profile state from JSON file.

    Args:
        file_path: Path to profile state JSON file.
                   Relative paths resolved from current working directory.

    Returns:
        Parsed and validated ProfileState model.

    Raises:
        ToolError: If file not found or invalid JSON/schema.
    """
    state_path = Path(file_path)
    if not state_path.is_absolute():
        state_path = Path.cwd() / file_path

    if not state_path.exists():
        raise fastmcp.exceptions.ToolError(f'Profile state file not found: {state_path}')

    profile_state_json = state_path.read_text()
    return ProfileState.model_validate_json(profile_state_json)


async def _inject_cookies_via_cdp(driver: webdriver.Chrome, cookies: Sequence[ProfileStateCookie]) -> int:
    """Inject cookies via CDP Network.setCookies.

    Cookies are set BEFORE navigation so they're sent with the request.
    CDP API requires camelCase field names, so we convert from our
    snake_case ProfileStateCookie fields.

    Args:
        driver: WebDriver instance with CDP support.
        cookies: Sequence of ProfileStateCookie objects.

    Returns:
        Number of cookies injected.
    """
    if not cookies:
        return 0

    cdp_cookies = []
    for cookie in cookies:
        cdp_cookie = {
            'name': cookie.name,
            'value': cookie.value,
            'domain': cookie.domain,
            'path': cookie.path,
            'httpOnly': cookie.http_only,
            'secure': cookie.secure,
            'sameSite': cookie.same_site,
        }
        # Handle session cookies (expires: -1) vs persistent cookies
        # For CDP, omit expires for session cookies
        if cookie.expires != -1:
            cdp_cookie['expires'] = cookie.expires

        cdp_cookies.append(cdp_cookie)

    # Set all cookies at once
    await asyncio.to_thread(driver.execute_cdp_cmd, 'Network.setCookies', {'cookies': cdp_cookies})

    return len(cdp_cookies)


async def _setup_pending_profile_state(service: BrowserService, profile_state: ProfileState) -> None:
    """Configure lazy restore for IndexedDB (localStorage/sessionStorage handled by init script).

    Stores profile state for lazy IndexedDB restoration as origins are visited.
    localStorage and sessionStorage are restored via Page.addScriptToEvaluateOnNewDocument
    which runs BEFORE page JavaScript - no lazy restore needed.

    IndexedDB requires JavaScript execution in page context with async APIs, so it
    must be restored lazily via _restore_pending_profile_state_for_current_origin.

    Args:
        service: BrowserService instance.
        profile_state: ProfileState to restore lazily (IndexedDB only).
    """
    service.state.pending_profile_state = profile_state
    service.state.restored_origins.clear()


async def _install_response_body_capture_if_needed(
    driver: webdriver.Chrome,
    service: BrowserService,
    enable_har_capture: bool,
    log_prefix: str,
) -> None:
    """Install JS interceptor for HAR response body capture if not already installed.

    The interceptor patches fetch() and XMLHttpRequest to capture response bodies
    in real-time, before Chrome garbage collects them. This enables export_har()
    to retrieve bodies for API calls that JavaScript consumes immediately.

    Args:
        driver: WebDriver instance to install script on.
        service: BrowserService to track installation state.
        enable_har_capture: Whether HAR capture was requested.
        log_prefix: Function name for log messages (e.g., 'navigate').
    """
    if enable_har_capture and not service.state.response_body_capture_enabled:
        await asyncio.to_thread(
            driver.execute_cdp_cmd,
            'Page.addScriptToEvaluateOnNewDocument',
            {'source': RESPONSE_BODY_CAPTURE_SCRIPT},
        )
        service.state.response_body_capture_enabled = True
        logger.info('Response body capture interceptor installed')


# -- CDP and storage helpers (shared by BrowserService methods) --


async def _download_with_browser_context(
    service: BrowserService, driver: webdriver.Chrome, url: str
) -> tuple[bytes, int, str]:
    """Download a URL using httpx with headers and cookies extracted from the browser session.

    Builds browser-realistic request headers (User-Agent, Referer, Sec-Fetch-*) and
    forwards domain-scoped cookies so CDNs see the request as coming from the same
    browser. Routes through mitmproxy when proxy is configured.
    """
    user_agent = await asyncio.to_thread(driver.execute_script, 'return navigator.userAgent')
    current_url = await asyncio.to_thread(lambda: driver.current_url)

    headers = {
        'User-Agent': user_agent,
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': current_url,
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'no-cors',
        'Sec-Fetch-Site': 'cross-site',
    }

    selenium_cookies = await asyncio.to_thread(driver.get_cookies)
    jar = httpx.Cookies()
    for cookie in selenium_cookies:
        jar.set(cookie['name'], cookie['value'], domain=cookie.get('domain', ''))

    proxy = 'http://127.0.0.1:8080' if service.state.proxy_config else None

    try:
        async with httpx.AsyncClient(
            cookies=jar,
            headers=headers,
            follow_redirects=True,
            timeout=60.0,
            proxy=proxy,
            verify=proxy is None,  # mitmproxy uses self-signed certs
        ) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise fastmcp.exceptions.ToolError(f'Download timed out after 60s: {url}') from exc
    except httpx.HTTPError as exc:
        raise fastmcp.exceptions.ToolError(f'Network error downloading {url}: {exc}') from exc

    content_type = response.headers.get('content-type', 'unknown')
    return response.content, response.status_code, content_type


async def _cdp_enable_domstorage(driver: webdriver.Chrome) -> None:
    """Enable CDP DOMStorage domain. Idempotent - safe to call multiple times."""
    await asyncio.to_thread(driver.execute_cdp_cmd, 'DOMStorage.enable', {})


async def _cdp_get_storage(driver: webdriver.Chrome, origin: str, is_local: bool = True) -> Sequence[Mapping[str, str]]:
    """Get storage items for a specific origin via CDP.

    Args:
        driver: WebDriver with CDP support
        origin: Security origin (e.g., "https://example.com")
        is_local: True for localStorage, False for sessionStorage

    Returns:
        Sequence of {name, value} dicts, or empty sequence if origin has no storage

    Note:
        - Requires DOMStorage domain to be enabled first
        - Read requires active frame for target origin (same as write)
        - This is why we pre-capture before navigation - departed origins have no frame
    """
    result = await asyncio.to_thread(
        driver.execute_cdp_cmd,
        'DOMStorage.getDOMStorageItems',
        {'storageId': {'securityOrigin': origin, 'isLocalStorage': is_local}},
    )
    # CDP returns {"entries": [["key1", "value1"], ["key2", "value2"], ...]}
    entries = result.get('entries', [])
    return [{'name': kv[0], 'value': kv[1]} for kv in entries]


async def _cdp_set_storage(driver: webdriver.Chrome, origin: str, key: str, value: str, is_local: bool = True) -> None:
    """Set a storage item for a specific origin via CDP.

    Args:
        driver: WebDriver with CDP support
        origin: Security origin (e.g., "https://example.com")
        key: Storage key
        value: Storage value
        is_local: True for localStorage, False for sessionStorage

    Raises:
        WebDriverException: "Frame not found for the given storage id" if no active frame

    Note:
        - Requires DOMStorage domain to be enabled first
        - Write REQUIRES active frame for target origin (frame-dependent)
        - This is why we use lazy restore pattern
    """
    await asyncio.to_thread(
        driver.execute_cdp_cmd,
        'DOMStorage.setDOMStorageItem',
        {
            'storageId': {'securityOrigin': origin, 'isLocalStorage': is_local},
            'key': key,
            'value': value,
        },
    )


async def _capture_current_origin_storage(service: BrowserService, driver: webdriver.Chrome) -> None:
    """Capture current origin's localStorage, sessionStorage, and IndexedDB to cache.

    Call this BEFORE any action that might navigate away (click, press_key, navigate).
    Safe to call multiple times - idempotent, overwrites previous cache for same origin.

    This is necessary because:
    - CDP DOMStorage requires an active frame for the origin
    - IndexedDB JavaScript capture requires page context
    Once you navigate away, the frame is gone and these operations fail.
    """
    current_url = driver.current_url
    if not current_url or current_url.startswith(('about:', 'data:', 'chrome:', 'blob:', 'file://')):
        return

    current_origin = await asyncio.to_thread(driver.execute_script, 'return window.location.origin')
    if not current_origin or current_origin == 'null':
        return

    await _cdp_enable_domstorage(driver)

    # Capture localStorage - always update cache (even if empty) to avoid stale data
    # if user clears storage and then navigates away
    local_storage_items = await _cdp_get_storage(driver, current_origin, is_local=True)
    service.state.local_storage_cache[current_origin] = [dict(item) for item in local_storage_items]

    # Capture sessionStorage - always update cache (even if empty)
    session_storage_items = await _cdp_get_storage(driver, current_origin, is_local=False)
    service.state.session_storage_cache[current_origin] = [dict(item) for item in session_storage_items]

    # Capture IndexedDB - requires JavaScript (CDP has no write API, so we use JS for both)
    # This may add ~100ms for sites with IndexedDB; most sites have none so this is fast.
    # Note: INDEXEDDB_CAPTURE_SCRIPT returns a Promise, so we must use execute_async_script
    # with a wrapper that awaits the Promise and calls the Selenium callback.
    # Wrap in IIFE so 'return' statement works, then chain .then() on the Promise.
    async_wrapper = f"""
        var callback = arguments[arguments.length - 1];
        (function() {{ {INDEXEDDB_CAPTURE_SCRIPT} }})()
            .then(function(r) {{ callback(r); }})
            .catch(function(e) {{ callback([]); }});
    """
    raw_result = await asyncio.to_thread(driver.execute_async_script, async_wrapper)
    indexeddb_result: list[dict[str, Any]] = raw_result if isinstance(raw_result, list) else []
    # INDEXEDDB_CAPTURE_SCRIPT returns list of database dicts, or empty list
    if indexeddb_result:
        service.state.indexed_db_cache[current_origin] = indexeddb_result
    else:
        # No databases - still update cache to avoid stale data
        service.state.indexed_db_cache[current_origin] = []

    # Log combined stats
    total_domstorage = len(local_storage_items) + len(session_storage_items)
    indexeddb_count = len(service.state.indexed_db_cache.get(current_origin, []))
    if total_domstorage > 0 or indexeddb_count > 0:
        parts = []
        if local_storage_items:
            parts.append(f'{len(local_storage_items)} localStorage')
        if session_storage_items:
            parts.append(f'{len(session_storage_items)} sessionStorage')
        if indexeddb_count > 0:
            parts.append(f'{indexeddb_count} IndexedDB databases')
        logger.info('Cached %s for %s', ' + '.join(parts), current_origin)


async def _restore_pending_profile_state_for_current_origin(service: BrowserService, driver: webdriver.Chrome) -> None:
    """Restore IndexedDB for current origin (localStorage/sessionStorage handled by init script).

    Called after navigation to check if we have pending IndexedDB data for the
    new current origin. Tracks restored origins to avoid double-restore which
    could overwrite page modifications.

    localStorage and sessionStorage are restored via Page.addScriptToEvaluateOnNewDocument
    (init script approach), which runs BEFORE page JavaScript. This ensures storage
    is populated before apps check for auth tokens or initialization data.

    IndexedDB requires JavaScript execution in page context and uses async APIs,
    so it must be restored after navigation via this lazy restore mechanism.

    Note: sessionStorage is session-scoped. Restoring it to a new browser context
    works, but the data will be lost when the browser closes (correct browser behavior).
    """
    if not service.state.pending_profile_state:
        return

    current_origin = await asyncio.to_thread(driver.execute_script, 'return window.location.origin')
    if not current_origin or current_origin == 'null':
        return

    # Skip special origins
    if current_origin.startswith(('chrome://', 'about:', 'data:', 'blob:', 'file://')):
        return

    if current_origin in service.state.restored_origins:
        return  # Already restored in this session

    # Find storage data for this origin (dict lookup, not array iteration)
    origin_data = service.state.pending_profile_state.origins.get(current_origin)

    if not origin_data:
        # Mark as checked so we don't repeatedly search for non-existent origin
        service.state.restored_origins.add(current_origin)
        return

    # Check if there's IndexedDB to restore
    # Note: localStorage/sessionStorage are handled by init script (runs before page JS)
    has_indexed_db = origin_data.indexed_db and len(origin_data.indexed_db) > 0

    if not has_indexed_db:
        service.state.restored_origins.add(current_origin)
        return

    # Restore IndexedDB via JavaScript (CDP has no write API)
    # Note: localStorage/sessionStorage handled by init script (runs before page JS)
    # Note: INDEXEDDB_RESTORE_SCRIPT returns a Promise, so we must use execute_async_script.
    # The script expects arguments[0] to contain the databases list, so we use .apply()
    # to set up the arguments array. Wrap in IIFE so 'return' statement works.
    indexeddb_count = 0
    if origin_data.indexed_db:  # Defensive check even though has_indexed_db was true
        # Use by_alias=True to serialize as camelCase for JavaScript compatibility
        databases_list = [db.model_dump(by_alias=True) for db in origin_data.indexed_db]
        databases_json = json.dumps(databases_list)
        async_wrapper = f"""
            var callback = arguments[arguments.length - 1];
            var data = {databases_json};
            (function() {{ {INDEXEDDB_RESTORE_SCRIPT} }}).apply(null, [data])
                .then(function(r) {{ callback(r); }})
                .catch(function(e) {{ callback({{success: false, error: String(e)}}); }});
        """
        restore_result = await asyncio.to_thread(driver.execute_async_script, async_wrapper)
        if restore_result and restore_result.get('success'):
            indexeddb_count = restore_result.get('databases_restored', 0)

    service.state.restored_origins.add(current_origin)

    if indexeddb_count > 0:
        logger.info('Restored %s IndexedDB databases for %s', indexeddb_count, current_origin)
