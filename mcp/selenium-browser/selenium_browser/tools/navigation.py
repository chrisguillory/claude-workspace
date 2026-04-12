from __future__ import annotations

__all__ = [
    'register_tools',
]

from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    Browser,
    NavigationResult,
)
from ..service import BrowserService


def register_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register navigation tools."""

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
        return await service.navigate(
            url=url,
            fresh_browser=fresh_browser,
            enable_har_capture=enable_har_capture,
            init_scripts=init_scripts,
            browser=browser,
        )

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
        return await service.navigate_with_profile_state(
            url=url,
            profile_state_file=profile_state_file,
            chrome_profile=chrome_profile,
            origins_filter=origins_filter,
            live_session_storage_via_applescript=live_session_storage_via_applescript,
            browser=browser,
            enable_har_capture=enable_har_capture,
            init_scripts=init_scripts,
        )
