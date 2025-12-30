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
from typing import Literal, Sequence
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
from src.scripts import (
    TEXT_EXTRACTION_SCRIPT,
    ARIA_SNAPSHOT_SCRIPT,
    NETWORK_MONITOR_SETUP_SCRIPT,
    NETWORK_MONITOR_CHECK_SCRIPT,
    WEB_VITALS_SCRIPT,
    RESOURCE_TIMING_SCRIPT,
    INDEXEDDB_CAPTURE_SCRIPT,
    INDEXEDDB_RESTORE_SCRIPT,
    build_execute_javascript_async_script,
)
from src.models import (
    # Chrome profiles
    ChromeProfilesResult,
    # Web Vitals
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
    # Network
    NetworkCapture,
    NetworkRequest,
    RequestTiming,
    # JavaScript execution
    JavaScriptResult,
    # Navigation and page extraction
    CapturedResource,
    ResourceCapture,
    HARExportResult,
    NavigationResult,
    InteractiveElement,
    FocusableElement,
    SmartExtractionInfo,
    PageTextResult,
    # Storage state (session persistence)
    StorageStateCookie,
    StorageStateLocalStorageItem,
    StorageStateOrigin,
    StorageState,
    SaveStorageStateResult,
)


class PrintLogger:
    """Simple logger that logs to stderr (MCP servers must not write to stdout)."""

    def __init__(self, ctx: Context | None = None):
        """Initialize logger. Context is ignored."""
        pass

    async def info(self, message: str):
        """Log info message to stderr."""
        print(f"[selenium-browser-automation] {message}", file=sys.stderr)


# =============================================================================
# CDP DOMStorage Helpers (Multi-Origin localStorage Access)
# =============================================================================


async def _cdp_enable_domstorage(driver: webdriver.Chrome) -> None:
    """Enable CDP DOMStorage domain. Idempotent - safe to call multiple times."""
    await asyncio.to_thread(driver.execute_cdp_cmd, "DOMStorage.enable", {})


async def _cdp_get_storage(
    driver: webdriver.Chrome,
    origin: str,
    is_local: bool = True,
) -> Sequence[dict]:
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
        "DOMStorage.getDOMStorageItems",
        {"storageId": {"securityOrigin": origin, "isLocalStorage": is_local}},
    )
    # CDP returns {"entries": [["key1", "value1"], ["key2", "value2"], ...]}
    entries = result.get("entries", [])
    return [{"name": kv[0], "value": kv[1]} for kv in entries]


async def _cdp_set_storage(
    driver: webdriver.Chrome,
    origin: str,
    key: str,
    value: str,
    is_local: bool = True,
) -> None:
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
        "DOMStorage.setDOMStorageItem",
        {
            "storageId": {"securityOrigin": origin, "isLocalStorage": is_local},
            "key": key,
            "value": value,
        },
    )


async def _capture_current_origin_storage(
    service: BrowserService,
    driver: webdriver.Chrome,
) -> None:
    """Capture current origin's localStorage and sessionStorage to cache before navigation.

    Call this BEFORE any action that might navigate away (click, press_key, navigate).
    Safe to call multiple times - idempotent, overwrites previous cache for same origin.

    This is necessary because CDP DOMStorage.getDOMStorageItems requires an active
    frame for the origin. Once you navigate away, the frame is gone and CDP returns
    "Frame not found" error.
    """
    current_url = driver.current_url
    if not current_url or current_url.startswith(("about:", "data:", "chrome:", "blob:", "file://")):
        return

    current_origin = await asyncio.to_thread(
        driver.execute_script, "return window.location.origin"
    )
    if not current_origin or current_origin == "null":
        return

    await _cdp_enable_domstorage(driver)

    # Capture localStorage - always update cache (even if empty) to avoid stale data
    # if user clears storage and then navigates away
    local_storage_items = await _cdp_get_storage(driver, current_origin, is_local=True)
    service.state.localStorage_cache[current_origin] = list(local_storage_items)

    # Capture sessionStorage - always update cache (even if empty)
    session_storage_items = await _cdp_get_storage(driver, current_origin, is_local=False)
    service.state.sessionStorage_cache[current_origin] = list(session_storage_items)

    # Log combined stats
    total_items = len(local_storage_items) + len(session_storage_items)
    if total_items > 0:
        print(
            f"[storage] Cached {len(local_storage_items)} localStorage + "
            f"{len(session_storage_items)} sessionStorage items for {current_origin}",
            file=sys.stderr,
        )


async def _restore_pending_storage_for_current_origin(
    service: BrowserService,
    driver: webdriver.Chrome,
) -> None:
    """Restore localStorage and sessionStorage for current origin from pending storage state.

    Called after navigation to check if we have pending storage for the
    new current origin. Tracks restored origins to avoid double-restore which
    could overwrite page modifications.

    CDP DOMStorage.setDOMStorageItem requires an active frame for the origin,
    so we can only restore storage AFTER navigating to each origin.
    This implements "lazy restore" - restore on-demand when we arrive at each origin.

    Note: sessionStorage is session-scoped. Restoring it to a new browser context
    works, but the data will be lost when the browser closes (correct browser behavior).
    """
    if not service.state.pending_storage_state:
        return

    current_origin = await asyncio.to_thread(
        driver.execute_script, "return window.location.origin"
    )
    if not current_origin or current_origin == "null":
        return

    # Skip special origins
    if current_origin.startswith(("chrome://", "about:", "data:", "blob:", "file://")):
        return

    if current_origin in service.state.restored_origins:
        return  # Already restored in this session

    # Find storage data for this origin
    origin_data = None
    for origin_entry in service.state.pending_storage_state.origins:
        if origin_entry.origin == current_origin:
            origin_data = origin_entry
            break

    if not origin_data:
        # Mark as checked so we don't repeatedly search for non-existent origin
        service.state.restored_origins.add(current_origin)
        return

    # Check if there's anything to restore
    has_localStorage = origin_data.localStorage and len(origin_data.localStorage) > 0
    has_sessionStorage = origin_data.sessionStorage and len(origin_data.sessionStorage) > 0

    if not has_localStorage and not has_sessionStorage:
        service.state.restored_origins.add(current_origin)
        return

    await _cdp_enable_domstorage(driver)

    # Restore localStorage
    local_count = 0
    if has_localStorage:
        for item in origin_data.localStorage:
            await _cdp_set_storage(driver, current_origin, item.name, item.value, is_local=True)
            local_count += 1

    # Restore sessionStorage
    session_count = 0
    if has_sessionStorage:
        for item in origin_data.sessionStorage:
            await _cdp_set_storage(driver, current_origin, item.name, item.value, is_local=False)
            session_count += 1

    service.state.restored_origins.add(current_origin)
    print(
        f"[storage] Restored {local_count} localStorage + {session_count} sessionStorage items for {current_origin}",
        file=sys.stderr,
    )


class OriginTracker:
    """Tracks origins visited during browser session for multi-origin storage capture.

    CDP storage APIs require explicit origin specification - they have no enumeration
    API (security by design). This tracker maintains a set of all origins visited
    via navigate() so save_storage_state() knows which origins to query.

    Origin format: scheme://host:port (e.g., "https://example.com", "http://localhost:8080")
    Port is included only if non-default (not 80 for http, not 443 for https).
    """

    def __init__(self) -> None:
        self._origins: set[str] = set()

    def add_origin(self, url: str) -> str:
        """Extract and track origin from URL. Returns the origin.

        Args:
            url: Full URL (e.g., "https://example.com/path?query=1")

        Returns:
            The extracted origin (e.g., "https://example.com")
        """
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        self._origins.add(origin)
        return origin

    def get_origins(self) -> list[str]:
        """Get sorted list of all tracked origins."""
        return sorted(self._origins)

    def clear(self) -> None:
        """Clear all tracked origins. Called on fresh_browser=True."""
        self._origins.clear()

    def __len__(self) -> int:
        return len(self._origins)


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
        # Proxy configuration for authenticated proxies
        self.proxy_config: dict[str, str] | None = None  # {host, port, username, password}
        self.mitmproxy_process: subprocess.Popen | None = None  # Local mitmproxy for upstream auth
        # Origin tracking for multi-origin storage capture
        self.origin_tracker = OriginTracker()
        # Storage caches - captured on navigate-away since CDP can't query departed origins
        self.localStorage_cache: dict[str, list[dict]] = {}
        self.sessionStorage_cache: dict[str, list[dict]] = {}
        # Lazy restore: pending storage state and tracking of already-restored origins
        self.pending_storage_state: StorageState | None = None
        self.restored_origins: set[str] = set()


class LoggerProtocol(typing.Protocol):
    """Protocol for logger - allows service to be MCP-agnostic."""

    async def info(self, message: str) -> None: ...


class BrowserService:
    """Browser automation service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: BrowserState) -> None:
        self.state = state  # Non-Optional - guaranteed by constructor

    async def close_browser(self) -> None:
        """Tear down browser and reset profile state.

        Also clears origin tracking since a fresh browser is a fresh session.
        """
        if self.state.driver:
            await asyncio.to_thread(self.state.driver.quit)
            self.state.driver = None
        self.state.current_profile = None
        # Clear origin tracking and storage caches - new browser = new session
        self.state.origin_tracker.clear()
        self.state.localStorage_cache.clear()
        self.state.sessionStorage_cache.clear()
        # Clear lazy restore state - new browser = fresh session
        self.state.pending_storage_state = None
        self.state.restored_origins.clear()

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
        init_scripts: Sequence[str] | None = None,
        storage_state_file: str | None = None,
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
            init_scripts: JavaScript code to run before every page load (requires fresh_browser=True).
                         Scripts persist for all navigations until next fresh_browser=True.
                         Use for API interceptors, environment patching.
            storage_state_file: Import storage state before navigation (requires fresh_browser=True).
                               Restores cookies and localStorage from Playwright-compatible JSON.
                               Use after save_storage_state() to restore authenticated sessions.

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

        Example - Session persistence:
            # After logging in:
            save_storage_state("auth.json")

            # Later, in new session:
            navigate(
                "https://example.com/account",
                fresh_browser=True,
                storage_state_file="auth.json"
            )
            # → Cookies restored before navigation, localStorage after
            # → Already authenticated!

        Note:
            init_scripts run BEFORE page scripts in every frame (including iframes).
            Do not modify navigator.webdriver, navigator.languages, navigator.plugins,
            or window.chrome - these are reserved for bot detection evasion.

            storage_state_file imports cookies before navigation (sent with request)
            and localStorage after navigation (requires origin context).
            Only localStorage for the target page's origin is restored.

        Returns:
            NavigationResult with current_url and title

        Next steps:
            1. get_aria_snapshot('body') - understand page structure
            2. get_interactive_elements() - find specific elements to click
            3. click(selector) - interact with elements

        For performance investigation:
            Use get_resource_timings() after navigation to measure page load timing.
            Use export_har() for detailed HTTP transaction data (requires enable_har_capture=True).
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

        if init_scripts and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                "init_scripts requires fresh_browser=True (scripts must be registered before first navigation)"
            )

        if storage_state_file and not fresh_browser:
            raise fastmcp.exceptions.ValidationError(
                "storage_state_file requires fresh_browser=True (import into clean session)"
            )

        print(
            f"[navigate] Navigating to {url}"
            + (" (fresh browser)" if fresh_browser else "")
            + (" (HAR capture enabled)" if enable_har_capture else "")
            + (f" ({len(init_scripts)} init scripts)" if init_scripts else "")
            + (" (importing storage state)" if storage_state_file else ""),
            file=sys.stderr,
        )

        if fresh_browser:
            await service.close_browser()

        driver = await service.get_browser(profile=profile, enable_har_capture=enable_har_capture)

        # Install user init scripts (after browser creation, before navigation)
        # Scripts registered here run on EVERY new document in this session
        if init_scripts:
            for script in init_scripts:
                await asyncio.to_thread(
                    driver.execute_cdp_cmd,
                    "Page.addScriptToEvaluateOnNewDocument",
                    {"source": script},
                )

        # Import storage state if provided (cookies BEFORE navigation, localStorage AFTER)
        storage_state: StorageState | None = None
        if storage_state_file:
            # Determine file path
            state_path = Path(storage_state_file)
            if not state_path.is_absolute():
                state_path = Path.cwd() / storage_state_file

            # Load and validate storage state file
            if not state_path.exists():
                raise fastmcp.exceptions.ToolError(
                    f"Storage state file not found: {state_path}"
                )

            storage_state_json = state_path.read_text()
            storage_state = StorageState.model_validate_json(storage_state_json)

            print(
                f"[navigate] Importing {len(storage_state.cookies)} cookies from {storage_state_file}",
                file=sys.stderr,
            )

            # Set cookies via CDP BEFORE navigation
            # This allows cookies to be sent with the navigation request
            if storage_state.cookies:
                # Convert storageState cookies to CDP format
                cdp_cookies = []
                for cookie in storage_state.cookies:
                    cdp_cookie = {
                        "name": cookie.name,
                        "value": cookie.value,
                        "domain": cookie.domain,
                        "path": cookie.path,
                        "httpOnly": cookie.httpOnly,
                        "secure": cookie.secure,
                        "sameSite": cookie.sameSite,
                    }
                    # Handle session cookies (expires: -1) vs persistent cookies
                    # For CDP, omit expires for session cookies
                    if cookie.expires != -1:
                        cdp_cookie["expires"] = cookie.expires

                    cdp_cookies.append(cdp_cookie)

                # Set all cookies at once
                await asyncio.to_thread(
                    driver.execute_cdp_cmd,
                    "Network.setCookies",
                    {"cookies": cdp_cookies}
                )

                print(
                    f"[navigate] Set {len(cdp_cookies)} cookies via CDP",
                    file=sys.stderr,
                )

        # PRE-ACTION: Capture localStorage before navigating away
        # (CDP can't query departed origins - frame is gone after navigation)
        await _capture_current_origin_storage(service, driver)

        # Navigate (blocking operation)
        await asyncio.to_thread(driver.get, url)

        # Track the final origin after redirects for multi-origin storage capture
        final_url = driver.current_url
        tracked_origin = service.state.origin_tracker.add_origin(final_url)

        print(
            f"[navigate] Successfully navigated to {final_url} "
            f"(tracked origins: {len(service.state.origin_tracker)})",
            file=sys.stderr,
        )

        # Lazy restore: store pending storage state if new file provided
        # CDP DOMStorage.setDOMStorageItem requires an active frame for the target origin,
        # so we can only restore localStorage as we visit each origin.
        if storage_state is not None:
            # Store for lazy restore - localStorage will be restored as we visit each origin
            service.state.pending_storage_state = storage_state
            service.state.restored_origins.clear()

        # ALWAYS try to restore localStorage for current origin (idempotent)
        # This handles both initial navigation with storage_state_file AND
        # subsequent navigations to other origins with pending storage state.
        # The helper checks restored_origins to avoid double-restore.
        await _restore_pending_storage_for_current_origin(service, driver)

        # IndexedDB restore is current-origin only (requires DOM context)
        if storage_state is not None:
            current_origin = await asyncio.to_thread(
                driver.execute_script,
                "return window.location.origin"
            )

            # Find current origin for IndexedDB restore
            origin_data: StorageStateOrigin | None = None
            for origin in storage_state.origins:
                if origin.origin == current_origin:
                    origin_data = origin
                    break

            # Restore IndexedDB if present in storage state
            if origin_data and origin_data.indexedDB:
                indexeddb_databases = [
                    db.model_dump() for db in origin_data.indexedDB
                ]

                print(
                    f"[navigate] Restoring {len(indexeddb_databases)} IndexedDB databases",
                    file=sys.stderr,
                )

                # Use the loaded script from src/scripts/indexeddb_restore.js
                restore_result = await asyncio.to_thread(
                    driver.execute_script,
                    INDEXEDDB_RESTORE_SCRIPT,
                    indexeddb_databases
                )

                if restore_result.get("success"):
                    print(
                        f"[navigate] IndexedDB restored: "
                        f"{restore_result.get('databases_restored', 0)} databases, "
                        f"{restore_result.get('records_restored', 0)} records",
                        file=sys.stderr,
                    )

                    # Log any errors encountered during restoration
                    if restore_result.get("errors"):
                        for error in restore_result["errors"][:3]:  # Limit to first 3
                            print(
                                f"[navigate] IndexedDB warning: {error}",
                                file=sys.stderr,
                            )
                else:
                    print(
                        f"[navigate] IndexedDB restoration failed: "
                        f"{restore_result.get('errors', ['Unknown error'])}",
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

        # Execute the extraction script (loaded from src/scripts/)
        try:
            result = await asyncio.to_thread(
                driver.execute_script, TEXT_EXTRACTION_SCRIPT, selector
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
            # Use CDP to capture full page
            result = await asyncio.to_thread(
                driver.execute_cdp_cmd,
                "Page.captureScreenshot",
                {"captureBeyondViewport": True},
            )

            if "data" not in result:
                raise fastmcp.exceptions.ToolError(
                    "CDP full-page capture returned no data. Use full_page=False for viewport screenshot."
                )

            screenshot_data = base64.b64decode(result["data"])
            screenshot_path.write_bytes(screenshot_data)
            await logger.info(f"Full-page screenshot saved to {screenshot_path}")
            return str(screenshot_path)

        # Viewport screenshot
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

        # Execute ARIA snapshot script (loaded from src/scripts/)
        snapshot_data = await asyncio.to_thread(
            driver.execute_script, ARIA_SNAPSHOT_SCRIPT, selector, include_urls
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
                    classes: el.getAttribute('class') || ''
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

        # PRE-ACTION: Capture localStorage before click (might navigate away)
        await _capture_current_origin_storage(service, driver)
        url_before = driver.current_url

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

        # POST-ACTION: Check if navigation occurred, track new origin and restore localStorage
        url_after = driver.current_url
        if url_after != url_before:
            service.state.origin_tracker.add_origin(url_after)
            await logger.info(f"Navigation detected: {url_before} -> {url_after}")
            # Lazy restore: if we navigated to a new origin with pending storage state
            await _restore_pending_storage_for_current_origin(service, driver)

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

        # Step 1: Inject monitoring script (loaded from src/scripts/)
        await asyncio.to_thread(driver.execute_script, NETWORK_MONITOR_SETUP_SCRIPT)
        await logger.info("Network monitor injected")

        # Step 2: Poll for idle state (500ms threshold)
        start_time = time.time()
        idle_threshold_ms = 500
        timeout_s = timeout / 1000

        while time.time() - start_time < timeout_s:
            status = await asyncio.to_thread(driver.execute_script, NETWORK_MONITOR_CHECK_SCRIPT)

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

        # PRE-ACTION: Capture localStorage before key press (ENTER might submit form and navigate)
        await _capture_current_origin_storage(service, driver)
        url_before = driver.current_url

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

        # POST-ACTION: Check if navigation occurred, track new origin and restore localStorage
        url_after = driver.current_url
        if url_after != url_before:
            service.state.origin_tracker.add_origin(url_after)
            await logger.info(f"Navigation detected: {url_before} -> {url_after}")
            # Lazy restore: if we navigated to a new origin with pending storage state
            await _restore_pending_storage_for_current_origin(service, driver)

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

        errors: list[str] = []

        # Use execute_async_script for Promise-based collection (script from src/scripts/)
        results = await asyncio.to_thread(
            driver.execute_async_script,
            WEB_VITALS_SCRIPT,
            timeout_ms,
        )

        collection_duration = (time.time() - start_time) * 1000

        # Parse results into Pydantic models
        def parse_metric(data, model_cls):
            if not data:
                return None
            if isinstance(data, dict) and "error" in data:
                errors.append(f"{data.get('name', 'Unknown')}: {data['error']}")
                return None
            return model_cls(**data)

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
    with many resources, use clear_resource_timing_buffer=True to prevent data loss."""
    )
    async def get_resource_timings(
        ctx: Context,
        clear_resource_timing_buffer: bool = False,
        min_duration_ms: int = 0,
    ) -> NetworkCapture:
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        start_time = time.time()
        current_url = driver.current_url

        await logger.info(f"Getting resource timings for {current_url}")

        # Collect resource timing via JavaScript Performance API (script from src/scripts/)
        raw_entries = await asyncio.to_thread(driver.execute_script, RESOURCE_TIMING_SCRIPT)

        # Handle empty/null response
        if not raw_entries:
            raw_entries = []

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
        if clear_resource_timing_buffer:
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
                    body_result = await asyncio.to_thread(
                        driver.execute_cdp_cmd,
                        "Network.getResponseBody",
                        {"requestId": request_id},
                    )
                    if body_result.get("body"):
                        har_entry["response"]["content"]["text"] = body_result["body"]
                        if body_result.get("base64Encoded"):
                            har_entry["response"]["content"]["encoding"] = "base64"

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
        logger = PrintLogger(ctx)
        driver = await service.get_browser()

        await logger.info(f"Executing JavaScript ({len(code)} chars)")

        # Escape user code as JSON string to prevent injection
        # json.dumps handles quotes, backslashes, newlines, etc.
        escaped_code = json.dumps(code)

        # Build the async wrapper for Selenium's execute_async_script
        # (uses safe_serialize.js loaded from src/scripts/)
        async_script = build_execute_javascript_async_script(escaped_code)

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

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Save Storage State",
            readOnlyHint=False,
            idempotentHint=False,
        )
    )
    async def save_storage_state(
        filename: str,
        include_indexeddb: bool = False,
        ctx: Context | None = None,
    ) -> SaveStorageStateResult:
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
            SaveStorageStateResult with path, cookie count, and metadata

        What's captured:
            - All cookies with full attributes (HttpOnly, Secure, SameSite, expires)
            - localStorage for all tracked origins (multi-origin via lazy capture)
            - sessionStorage for all tracked origins (multi-origin via lazy capture)
            - IndexedDB databases (if include_indexeddb=True, current origin only)

        sessionStorage behavior:
            sessionStorage is session-scoped by browser design. Restored sessionStorage
            persists only for the lifetime of the browser context - closing the browser
            clears it. For cross-session persistence, use localStorage or cookies.

        Workflow:
            1. navigate("https://example.com/login", fresh_browser=True)
            2. [Complete login flow - click buttons, enter credentials, etc.]
            3. navigate("https://example.com/account")  # Navigate to authenticated page
            4. save_storage_state("example_auth.json", include_indexeddb=True)  # Export auth
            5. [Later, in new session:]
               navigate("https://example.com/account",
                       fresh_browser=True,
                       storage_state_file="example_auth.json")  # Restore auth

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
            - IndexedDB only captured for current origin (not multi-origin yet)
            - Tokens may expire between save and restore - re-authenticate if needed
        """
        logger = PrintLogger(ctx)

        driver = service.state.driver
        if driver is None:
            raise fastmcp.exceptions.ToolError(
                "Browser not initialized. Call navigate() first to establish a session."
            )

        await logger.info(f"Exporting storage state to {filename}")

        # Get all cookies via CDP Network.getCookies
        cookies_result = await asyncio.to_thread(
            driver.execute_cdp_cmd,
            "Network.getCookies",
            {}  # Empty = get all cookies
        )

        cdp_cookies = cookies_result.get("cookies", [])

        # Convert CDP cookies to storageState format
        storage_cookies: list[dict] = []
        for cookie in cdp_cookies:
            # CDP returns `session: true` for session cookies, otherwise `expires` timestamp
            # For storageState format: session cookies have expires=-1
            is_session = cookie.get("session", False)
            expires = -1.0 if is_session else cookie.get("expires", -1.0)

            # Normalize sameSite - CDP should return "Strict", "Lax", or "None"
            # but handle edge cases
            same_site = cookie.get("sameSite", "Lax")
            if same_site not in ("Strict", "Lax", "None"):
                same_site = "Lax"  # Default fallback

            storage_cookies.append({
                "name": cookie["name"],
                "value": cookie["value"],
                "domain": cookie["domain"],
                "path": cookie["path"],
                "expires": expires,
                "httpOnly": cookie.get("httpOnly", False),
                "secure": cookie.get("secure", False),
                "sameSite": same_site,
            })

        # Get current origin (for result metadata)
        current_origin = await asyncio.to_thread(
            driver.execute_script,
            "return window.location.origin"
        )

        # =================================================================
        # Multi-Origin Storage Capture (localStorage + sessionStorage)
        # =================================================================
        # CDP DOMStorage.getDOMStorageItems requires an active frame for the origin.
        # For departed origins (navigated away), we use the cached data captured
        # during navigate(). For the current origin, we query CDP directly.
        # =================================================================

        tracked_origins = service.state.origin_tracker.get_origins()
        await _cdp_enable_domstorage(driver)

        origins_data: list[dict] = []
        total_localstorage_items = 0
        total_sessionstorage_items = 0

        for origin in tracked_origins:
            # Skip special origins
            if origin.startswith(("chrome://", "about:", "data:", "blob:", "file://")):
                continue

            # Capture localStorage: cache for departed origins, CDP for current
            if origin == current_origin:
                local_storage_items = await _cdp_get_storage(driver, origin, is_local=True)
            elif origin in service.state.localStorage_cache:
                local_storage_items = service.state.localStorage_cache[origin]
            else:
                local_storage_items = []

            # Capture sessionStorage: cache for departed origins, CDP for current
            if origin == current_origin:
                session_storage_items = await _cdp_get_storage(driver, origin, is_local=False)
            elif origin in service.state.sessionStorage_cache:
                session_storage_items = service.state.sessionStorage_cache[origin]
            else:
                session_storage_items = []

            # Only add origin if it has any storage
            if local_storage_items or session_storage_items:
                origin_entry: dict = {
                    "origin": origin,
                    "localStorage": local_storage_items,
                }
                if session_storage_items:
                    origin_entry["sessionStorage"] = session_storage_items

                origins_data.append(origin_entry)
                total_localstorage_items += len(local_storage_items)
                total_sessionstorage_items += len(session_storage_items)

        await logger.info(
            f"Captured storage: {total_localstorage_items} localStorage + "
            f"{total_sessionstorage_items} sessionStorage items across "
            f"{len(origins_data)} origins (of {len(tracked_origins)} tracked)"
        )

        # Capture IndexedDB databases if requested
        indexeddb_data: list[dict] | None = None
        indexeddb_databases_count = 0
        indexeddb_records_count = 0

        if include_indexeddb:
            await logger.info("Capturing IndexedDB databases...")

            # Use the loaded script from src/scripts/indexeddb_capture.js
            indexeddb_data = await asyncio.to_thread(
                driver.execute_script, INDEXEDDB_CAPTURE_SCRIPT
            )

            # Handle null/undefined from JS
            if indexeddb_data is None:
                indexeddb_data = []

            # Count databases and records
            indexeddb_databases_count = len(indexeddb_data)
            for db in indexeddb_data:
                for store in db.get("objectStores", []):
                    indexeddb_records_count += len(store.get("records", []))

            await logger.info(
                f"Captured {indexeddb_databases_count} IndexedDB databases "
                f"with {indexeddb_records_count} total records"
            )

        # Add IndexedDB data to current origin if captured
        # Note: IndexedDB capture is still single-origin (current page) - Phase 3 will add multi-origin
        if include_indexeddb and indexeddb_data:
            # Find current origin in origins_data and add IndexedDB
            origin_found = False
            for origin_entry in origins_data:
                if origin_entry["origin"] == current_origin:
                    origin_entry["indexedDB"] = indexeddb_data
                    origin_found = True
                    break

            # If current origin had no localStorage/sessionStorage but has IndexedDB, add it
            if not origin_found:
                # Query sessionStorage for this edge case too
                current_session_storage = await _cdp_get_storage(driver, current_origin, is_local=False)
                origin_entry_new: dict = {
                    "origin": current_origin,
                    "localStorage": [],
                    "indexedDB": indexeddb_data,
                }
                if current_session_storage:
                    origin_entry_new["sessionStorage"] = list(current_session_storage)
                origins_data.append(origin_entry_new)

        # Build storageState structure with all captured origins
        storage_state_data = {
            "cookies": storage_cookies,
            "origins": origins_data,
        }

        # Validate with Pydantic before saving
        validated_state = StorageState(**storage_state_data)

        # Determine file path (allow absolute or relative to cwd)
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = Path.cwd() / filename

        # Save to file
        json_content = validated_state.model_dump_json(indent=2)
        file_path.write_text(json_content)

        result = SaveStorageStateResult(
            path=str(file_path),
            cookies_count=len(storage_cookies),
            origins_count=len(origins_data),
            current_origin=current_origin,
            size_bytes=len(json_content),
            indexeddb_databases_count=indexeddb_databases_count if include_indexeddb else None,
            indexeddb_records_count=indexeddb_records_count if include_indexeddb else None,
            tracked_origins=tracked_origins,
        )

        # Build log message
        log_parts = [
            f"Saved {result.cookies_count} cookies",
            f"{total_localstorage_items} localStorage + {total_sessionstorage_items} sessionStorage items "
            f"across {len(origins_data)} origins",
        ]
        if include_indexeddb and indexeddb_databases_count > 0:
            log_parts.append(f"{indexeddb_databases_count} IndexedDB databases ({indexeddb_records_count} records)")

        await logger.info(
            f"{' + '.join(log_parts)} for {current_origin} "
            f"to {file_path} ({result.size_bytes} bytes)"
        )

        return result


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
