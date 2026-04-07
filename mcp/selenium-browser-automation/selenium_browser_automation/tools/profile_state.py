from __future__ import annotations

__all__ = [
    'register_profile_state_tools',
]

from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    ChromeProfilesResult,
    ChromeProfileStateExportResult,
    SaveProfileStateResult,
)
from ..service import BrowserService


def register_profile_state_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register profile state tools."""

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

        return await service.save_profile_state(filename=filename, include_indexeddb=include_indexeddb)

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
