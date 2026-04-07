"""
Selenium Browser Automation MCP Server

CDP stealth injection to bypass Cloudflare bot detection.

Install:
    uvx --from git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/selenium-browser-automation mcp-selenium-browser-automation-server

Architecture: Runs locally (not Docker) for visible browser monitoring.
Uses Selenium with CDP stealth injection where Playwright fails.
"""

from __future__ import annotations

# Standard Library
import asyncio
import contextlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import types
import typing
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlparse

# Third-Party Libraries
import fastmcp.exceptions
import httpx
from cc_lib.types import JsonObject
from cc_lib.utils import Timer
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

# Local
from . import chrome_profile_state_export

# Local imports
from .helpers import build_storage_init_script
from .models import (
    # Browser selection
    Browser,
    # Navigation and page extraction
    ChromeProfilesResult,
    # Chrome profile state export
    ChromeProfileStateExportResult,
    ClearProxyResult,
    ConfigureProxyResult,
    # Console logs
    ConsoleLogsResult,
    # Web Vitals
    CoreWebVitals,
    DownloadResourceResult,
    FocusableElement,
    HARExportResult,
    InteractiveElement,
    # JavaScript execution
    JavaScriptResult,
    NavigationResult,
    # Network
    NetworkCapture,
    PageTextResult,
    ProfileState,
    # Profile state (browser state persistence)
    ProfileStateCookie,
    ProfileStateIndexedDB,
    ProfileStateOriginStorage,
    ResizeWindowResult,
    SaveProfileStateResult,
    SetBlockedURLsResult,
    SleepResult,
    WaitForSelectorResult,
)
from .scripts import (
    INDEXEDDB_CAPTURE_SCRIPT,
    INDEXEDDB_RESTORE_SCRIPT,
    RESPONSE_BODY_CAPTURE_SCRIPT,
)
from .service import BrowserService
from .state import BrowserState
from .validators import validate_css_selector as _validate_css_selector_impl

__all__ = [
    'lifespan',
    'main',
    'register_tools',
]

# Valid URL prefixes for navigation (navigate, navigate_with_profile_state).
VALID_URL_PREFIXES = ('http://', 'https://', 'file://', 'about:', 'data:', 'blob:')

# Large output threshold for file saving.
# When tool output exceeds this, save to file to preserve line structure.
# Matches python-interpreter MCP server for consistency.
# Without this, Claude Code's JSON-wrapped file saving escapes newlines,
# making Read tool's line-based offset/limit useless.
LARGE_OUTPUT_THRESHOLD = 25_000  # characters

# Hidden reason key mappings for metadata footer (JS key → display label).
# Different for ARIA vs visual tree because they filter on different criteria.
_ARIA_HIDDEN_REASON_KEYS: Sequence[tuple[str, str]] = [
    ('ariaHidden', 'aria-hidden'),
    ('displayNone', 'display-none'),
    ('visibilityHidden', 'visibility-hidden'),
    ('inert', 'inert'),
    ('other', 'other'),
]

_VISUAL_HIDDEN_REASON_KEYS: Sequence[tuple[str, str]] = [
    ('displayNone', 'display-none'),
    ('visibilityHidden', 'visibility-hidden'),
    ('opacity', 'opacity'),
    ('clipped', 'clipped'),
    ('offscreen', 'offscreen'),
    ('other', 'other'),
]


def register_tools(service: BrowserService) -> None:
    """Register service methods as MCP tools via closures."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Navigate to URL',
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def navigate(
        url: str,
        fresh_browser: bool = False,
        enable_har_capture: bool = False,
        init_scripts: Sequence[str] | None = None,
        browser: Browser | None = None,
        ctx: Context[Any, Any, Any] | None = None,
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
            browser = service.state.current_browser or 'chromium'

        if not url.startswith(VALID_URL_PREFIXES):
            raise fastmcp.exceptions.ValidationError(
                'URL must start with http://, https://, file://, about:, data:, or blob:',
            )

        if service.state.current_browser is not None and browser != service.state.current_browser and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                f"Browser changed from '{service.state.current_browser}' to '{browser}'. "
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
            await service.close_browser()

        driver = await service.get_browser(enable_har_capture=enable_har_capture, browser=browser)

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
        await _install_response_body_capture_if_needed(driver, service, enable_har_capture, 'navigate')

        # PRE-ACTION: Capture localStorage before navigating away
        # (CDP can't query departed origins - frame is gone after navigation)
        await _capture_current_origin_storage(service, driver)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        # Track the final origin after redirects for multi-origin storage capture
        final_url = driver.current_url
        service.state.origin_tracker.add_origin(final_url)

        logger.info('Successfully navigated to %s (tracked origins: %s)', final_url, len(service.state.origin_tracker))

        # Lazy restore: if navigate_with_profile_state() was called previously, restore
        # storage for current origin. This handles multi-origin sessions where the user
        # navigates to different origins after the initial session import.
        # The helper is idempotent and checks restored_origins to avoid double-restore.
        await _restore_pending_profile_state_for_current_origin(service, driver)

        return NavigationResult(current_url=driver.current_url, title=driver.title)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Navigate with Profile State',
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    async def navigate_with_profile_state(
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
        ctx: Context[Any, Any, Any] | None = None,
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
            browser = service.state.current_browser or 'chromium'

        # Always start fresh browser for session import
        await service.close_browser()

        # Get browser with configuration
        driver = await service.get_browser(enable_har_capture=enable_har_capture, browser=browser)

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
            service,
            enable_har_capture,
            'navigate_with_profile_state',
        )

        # Inject cookies via CDP BEFORE navigation
        cookies_injected = await _inject_cookies_via_cdp(driver, profile_state.cookies)

        logger.info('Injected %s cookies via CDP', cookies_injected)

        # PRE-ACTION: Capture localStorage before navigating away
        await _capture_current_origin_storage(service, driver)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        # Track the final origin after redirects
        final_url = driver.current_url
        service.state.origin_tracker.add_origin(final_url)

        logger.info('Successfully navigated to %s (tracked origins: %s)', final_url, len(service.state.origin_tracker))

        # Setup lazy restore for localStorage/sessionStorage/IndexedDB
        await _setup_pending_profile_state(service, profile_state)

        # Restore storage for current origin immediately
        await _restore_pending_profile_state_for_current_origin(service, driver)

        return NavigationResult(
            current_url=driver.current_url,
            title=driver.title,
            elapsed_seconds=round(timer.elapsed(), 3),
        )

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

    @mcp.tool(annotations=ToolAnnotations(title='Take Screenshot', readOnlyHint=True))
    async def screenshot(filename: str, ctx: Context[Any, Any, Any], full_page: bool = False) -> str:
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
        return await service.screenshot(filename=filename, full_page=full_page)

    async def _download_with_browser_context(driver: webdriver.Chrome, url: str) -> tuple[bytes, int, str]:
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

    @mcp.tool(annotations=ToolAnnotations(title='Download Specific Resource', readOnlyHint=False, idempotentHint=False))
    async def download_resource(url: str, output_filename: str) -> DownloadResourceResult:
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
        return await service.download_resource(url=url, output_filename=output_filename)

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

    @mcp.tool(annotations=ToolAnnotations(title='Get Interactive Elements', readOnlyHint=True))
    async def get_interactive_elements(
        selector_scope: str,
        text_contains: str | None,
        tag_filter: Sequence[str] | None,
        limit: int | None,
        ctx: Context[Any, Any, Any],
    ) -> Sequence[InteractiveElement]:
        """Find clickable elements by text or other filters. Returns CSS selectors for click().

        This is the tool for finding elements by text content.
        Example: get_interactive_elements(text_contains="Continue") → returns CSS selector for click()

        Args:
            selector_scope: CSS selector to limit search (e.g., ".wizard" or "body")
            text_contains: Filter by text content (case-insensitive, None = no filter)
            tag_filter: Only specific tags (e.g., ["button", "a"], None = all tags)
            limit: Max results to return (None = unlimited)
            ctx: MCP context

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
        return await service.get_interactive_elements(
            selector_scope=selector_scope, text_contains=text_contains, tag_filter=tag_filter, limit=limit
        )

    @mcp.tool(annotations=ToolAnnotations(title='Get Focusable Elements', readOnlyHint=True))
    async def get_focusable_elements(only_tabbable: bool, ctx: Context[Any, Any, Any]) -> Sequence[FocusableElement]:
        """Get keyboard-navigable elements sorted by tab order.

        Args:
            only_tabbable: True = Tab key only (tabindex >= 0)
                          False = includes programmatic focus (tabindex >= -1)
            ctx: MCP context

        Returns:
            list[FocusableElement]: Sorted by tab order, each with tag, text, selector, tab_index
        """
        return await service.get_focusable_elements(only_tabbable=only_tabbable)

    @mcp.tool(annotations=ToolAnnotations(title='Click Element', destructiveHint=False, idempotentHint=False))
    async def click(
        css_selector: str,
        ctx: Context[Any, Any, Any],
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
        return await service.click(
            css_selector=css_selector, wait_for_network=wait_for_network, network_timeout=network_timeout
        )

    @mcp.tool(annotations=ToolAnnotations(title='Wait for Network Idle', readOnlyHint=True, idempotentHint=True))
    async def wait_for_network_idle(ctx: Context[Any, Any, Any], timeout: int = 10000) -> None:
        """Wait for network activity to settle after clicks or dynamic content loads.

        Uses JavaScript instrumentation to monitor Fetch and XMLHttpRequest activity.
        Waits for no active requests and 500ms of idle time.

        Args:
            ctx: MCP context
            timeout: Timeout in milliseconds (default 10000ms)

        Note: Uses JavaScript instrumentation of Fetch/XHR APIs to track network activity.
              Waits for no active requests + 500ms idle threshold.
        """
        return await service.wait_for_network_idle(timeout=timeout)

    @mcp.tool(annotations=ToolAnnotations(title='Press Keyboard Key', destructiveHint=False, idempotentHint=False))
    async def press_key(key: str, ctx: Context[Any, Any, Any]) -> None:
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
        return await service.press_key(key=key)

    @mcp.tool(annotations=ToolAnnotations(title='Type Text', destructiveHint=False, idempotentHint=False))
    async def type_text(text: str, ctx: Context[Any, Any, Any], delay_ms: int = 0) -> None:
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
        return await service.type_text(text=text, delay_ms=delay_ms)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Hover Over Element',
            destructiveHint=False,
            idempotentHint=True,
        ),
    )
    async def hover(css_selector: str, ctx: Context[Any, Any, Any], duration_ms: int = 0) -> None:
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
        return await service.hover(css_selector=css_selector, duration_ms=duration_ms)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Scroll Page',
            destructiveHint=False,
            idempotentHint=False,
        ),
    )
    async def scroll(
        ctx: Context[Any, Any, Any],
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
        return await service.scroll(
            direction=direction,
            scroll_amount=scroll_amount,
            css_selector=css_selector,
            behavior=behavior,
            position=position,
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Sleep',
            readOnlyHint=True,
            idempotentHint=True,
        ),
    )
    async def sleep(duration_ms: int, ctx: Context[Any, Any, Any], reason: str | None = None) -> SleepResult:
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
        return await service.sleep(duration_ms=duration_ms, reason=reason)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Wait for Selector',
            readOnlyHint=True,
            idempotentHint=False,  # State can change between calls
        ),
    )
    async def wait_for_selector(
        css_selector: str,
        ctx: Context[Any, Any, Any],
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
        return await service.wait_for_selector(css_selector=css_selector, state=state, timeout=timeout)

    @mcp.tool(annotations=ToolAnnotations(title='List Chrome Profiles', readOnlyHint=True, idempotentHint=True))
    async def list_chrome_profiles(
        verbose: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
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
        return await service.list_chrome_profiles(verbose=verbose)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Resize Browser Window',
            destructiveHint=False,
            idempotentHint=True,
        ),
    )
    async def resize_window(width: int, height: int, ctx: Context[Any, Any, Any]) -> ResizeWindowResult:
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
        return await service.resize_window(width=width, height=height)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Capture Core Web Vitals',
            readOnlyHint=True,
            idempotentHint=True,
        ),
    )
    async def capture_web_vitals(ctx: Context[Any, Any, Any], timeout_ms: int = 5000) -> CoreWebVitals:
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
        return await service.capture_web_vitals(timeout_ms=timeout_ms)

    @mcp.tool(
        description="""Get resource timing data from the browser's Performance API.

Returns timing breakdown for all resources loaded by the page. No setup required -
the Performance API is always available. Useful for identifying slow resources
and network bottlenecks.

Args:
    clear_resource_timing_buffer: If True, clears the timing buffer after retrieval.
        Default False (non-destructive). Set True when measuring sequential page loads
        to avoid mixing entries from previous pages.
    min_duration_ms: Only include requests slower than this threshold (0 = all).

Returns:
    NetworkCapture with requests, timing breakdown, and summary statistics
    including slowest requests and breakdown by resource type.

Note:
    Browsers maintain a buffer of 150-250 entries. For long sessions or pages
    with many resources, use clear_resource_timing_buffer=True to prevent data loss.""",
    )
    async def get_resource_timings(
        ctx: Context[Any, Any, Any],
        clear_resource_timing_buffer: bool = False,
        min_duration_ms: int = 0,
    ) -> NetworkCapture:
        return await service.get_resource_timings(
            clear_resource_timing_buffer=clear_resource_timing_buffer, min_duration_ms=min_duration_ms
        )

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
    4. Read the HAR file or import into Chrome DevTools""",
    )
    async def export_har(
        ctx: Context[Any, Any, Any],
        filename: str,
        include_response_bodies: bool = False,
        max_body_size_mb: int = 10,
    ) -> HARExportResult:
        driver = await service.get_browser()

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
            har_path = service.state.capture_dir / filename
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
                    if body_source is None and service.state.response_body_capture_enabled:
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
        har_path = service.state.capture_dir / filename
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

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Get Console Logs',
            readOnlyHint=True,
            idempotentHint=False,  # Clears buffer after retrieval
        ),
    )
    async def get_console_logs(
        ctx: Context[Any, Any, Any],
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
        return await service.get_console_logs(level_filter=level_filter, pattern=pattern)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Configure Proxy',
            destructiveHint=False,
            idempotentHint=False,
        ),
    )
    async def configure_proxy(
        host: str,
        port: int,
        username: str,
        password: str,
        ctx: Context[Any, Any, Any] | None = None,
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
        upstream_url = f'http://{host}:{port}'
        upstream_auth = f'{username}:{password}'

        try:
            service.state.mitmproxy_process = subprocess.Popen(
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
            if service.state.mitmproxy_process.poll() is not None:
                stderr = (
                    service.state.mitmproxy_process.stderr.read().decode()
                    if service.state.mitmproxy_process.stderr
                    else ''
                )
                raise RuntimeError(f'mitmproxy failed to start: {stderr}')

            logger.info('mitmproxy started on localhost:8080')

        except FileNotFoundError:
            service.state.proxy_config = None
            raise fastmcp.exceptions.ToolError('mitmproxy not found. Install with: pip install mitmproxy') from None
        except Exception as e:
            service.state.proxy_config = None
            if service.state.mitmproxy_process:
                service.state.mitmproxy_process.kill()
                service.state.mitmproxy_process = None
            raise fastmcp.exceptions.ToolError(f'Failed to start mitmproxy: {e}') from e

        return ConfigureProxyResult(
            status='proxy_configured',
            host=host,
            port=port,
            note='Browser will use this proxy on next navigate()',
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Clear Proxy',
            destructiveHint=False,
            idempotentHint=True,
        ),
    )
    async def clear_proxy(ctx: Context[Any, Any, Any] | None = None) -> ClearProxyResult:
        """Clear proxy configuration and return to direct connection.

        Stops the mitmproxy subprocess and clears proxy settings.

        Returns:
            Status dict confirming proxy cleared
        """
        return await service.clear_proxy()

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Set Blocked URLs',
            destructiveHint=False,
            idempotentHint=True,
        ),
    )
    async def set_blocked_urls(urls: Sequence[str], ctx: Context[Any, Any, Any] | None = None) -> SetBlockedURLsResult:
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
        return await service.set_blocked_urls(urls=urls)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Execute JavaScript',
            readOnlyHint=False,
            idempotentHint=False,
        ),
    )
    async def execute_javascript(code: str, ctx: Context[Any, Any, Any], timeout_ms: int = 5000) -> JavaScriptResult:
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
        return await service.execute_javascript(code=code, timeout_ms=timeout_ms)

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Save Profile State',
            readOnlyHint=False,
            idempotentHint=False,
        ),
    )
    async def save_profile_state(
        filename: str,
        include_indexeddb: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
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

        driver = service.state.driver
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

        tracked_origins = service.state.origin_tracker.get_origins()
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
            elif origin in service.state.local_storage_cache:
                local_storage_items = service.state.local_storage_cache[origin]
            else:
                local_storage_items = []

            # Capture sessionStorage: cache for departed origins, CDP for current
            if origin == current_origin:
                session_storage_items = await _cdp_get_storage(driver, origin, is_local=False)
            elif origin in service.state.session_storage_cache:
                session_storage_items = service.state.session_storage_cache[origin]
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
                elif origin in service.state.indexed_db_cache:
                    indexeddb_databases = service.state.indexed_db_cache[origin]

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

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Export Chrome Profile State',
            readOnlyHint=True,
            idempotentHint=True,
        ),
    )
    async def export_chrome_profile_state(
        output_file: str,
        chrome_profile: str = 'Default',
        include_session_storage: bool = True,
        include_indexeddb: bool = False,
        origins_filter: Sequence[str] | None = None,
        live_session_storage_via_applescript: bool = False,
        ctx: Context[Any, Any, Any] | None = None,
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

        return await service.export_chrome_profile_state(
            output_file=output_file,
            chrome_profile=chrome_profile,
            include_session_storage=include_session_storage,
            include_indexeddb=include_indexeddb,
            origins_filter=origins_filter,
            live_session_storage_via_applescript=live_session_storage_via_applescript,
        )


@contextlib.asynccontextmanager
async def lifespan(server_instance: FastMCP) -> typing.AsyncIterator[None]:
    """Manage browser lifecycle - initialization before requests, cleanup after shutdown."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )

    state = await BrowserState.create()
    service = BrowserService(state)
    register_tools(service)

    # Register signal handlers to ensure cleanup on SIGTERM/SIGINT
    # This is critical for `claude mcp reconnect` which sends SIGTERM
    def signal_handler(signum: int, frame: types.FrameType | None) -> None:
        _sync_cleanup(state)
        sys.stderr.flush()
        os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info('Browser service initialized (screenshots: %s, captures: %s)', state.screenshot_dir, state.capture_dir)

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
    logger.info('Server cleanup complete')


mcp = FastMCP('selenium-browser-automation', lifespan=lifespan)


def main() -> None:
    """Entry point for uvx installation."""
    logger.info('Starting Selenium Browser Automation MCP server (CDP stealth injection for bot detection bypass)')
    mcp.run()


def _validate_css_selector(selector: str) -> None:
    """Validate CSS selector, converting ValueError to ToolError for MCP layer."""
    try:
        _validate_css_selector_impl(selector)
    except ValueError as e:
        raise fastmcp.exceptions.ToolError(str(e)) from None


logger = logging.getLogger(__name__)


# -- CDP DOMStorage Helpers (Multi-Origin localStorage Access) -----------------


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


# -- Profile State Import Helpers ----------------------------------------------


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


def _sync_cleanup(state: BrowserState, timeout: int = 5) -> None:
    """Synchronous cleanup for signal handlers (runs in main thread).

    Uses a thread with timeout to prevent hanging on unresponsive ChromeDriver.
    """
    print('\n⚠ Signal received, cleaning up browser...', file=sys.stderr)
    if state.driver:
        try:
            # driver.quit() sends an HTTP request to ChromeDriver which can hang
            # if Chrome/ChromeDriver is unresponsive. Run in a thread with timeout.
            quit_thread = threading.Thread(target=state.driver.quit, daemon=True)
            quit_thread.start()
            quit_thread.join(timeout=timeout)
            if quit_thread.is_alive():
                print(f'✗ Browser close timed out after {timeout}s, abandoning', file=sys.stderr)
            else:
                print('✓ Browser closed', file=sys.stderr)
        except Exception as e:  # exception_safety_linter.py: swallowed-exception — signal cleanup must not raise
            print(f'✗ Browser close error: {e}', file=sys.stderr)
    if state.mitmproxy_process:
        state.mitmproxy_process.terminate()
        try:
            state.mitmproxy_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            state.mitmproxy_process.kill()
    try:
        state.temp_dir.cleanup()
        state.capture_temp_dir.cleanup()
    except Exception:  # exception_safety_linter.py: swallowed-exception — signal cleanup must not raise
        pass
    print('✓ Signal cleanup complete, exiting', file=sys.stderr)


if __name__ == '__main__':
    main()
