"""Live sessionStorage extraction from Chrome via AppleScript (macOS only).

This module provides the ability to extract current, in-memory sessionStorage
from running Chrome tabs using AppleScript. This solves the problem of stale
sessionStorage data when reading from Chrome's LevelDB files on disk.

Requirement:
    Chrome > View > Developer > Allow JavaScript from Apple Events (one-time setting)

Platform:
    macOS only. On other platforms, functions return appropriate failure results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse


class SeleniumChromeConflictError(Exception):
    """Raised when Selenium Chrome is running and would conflict with AppleScript targeting."""

    pass


@dataclass
class AppleScriptExtractionResult:
    """Result from attempting AppleScript sessionStorage extraction."""

    success: bool
    session_storage: Mapping[str, Mapping[str, str]]  # origin -> {key: value}
    source: Literal['live', 'unavailable']
    error_reason: str | None = None
    tabs_matched: int = 0
    tabs_extracted: int = 0


def is_applescript_available() -> bool:
    """Check if AppleScript extraction is available on this system.

    Returns True only if:
    - Running on macOS (darwin)
    - Chrome is installed

    Does NOT check if the JavaScript setting is enabled (requires runtime test).
    """
    if sys.platform != 'darwin':
        return False

    chrome_path = Path('/Applications/Google Chrome.app')
    return chrome_path.exists()


def is_chrome_running() -> bool:
    """Check if Chrome is currently running."""
    if sys.platform != 'darwin':
        return False

    result = subprocess.run(
        ['pgrep', '-x', 'Google Chrome'],
        capture_output=True,
    )
    return result.returncode == 0


def is_selenium_chrome_running() -> bool:
    """Check if a Selenium-controlled Chrome is running.

    Selenium Chrome is identified by:
    - --test-type=webdriver flag
    - --user-data-dir pointing to temp directory (/var/folders/ or /tmp/)

    Returns True if Selenium Chrome detected, False otherwise.
    """
    if sys.platform != 'darwin':
        return False

    # Get all Chrome main processes with full command lines
    result = subprocess.run(
        ['ps', 'auxww'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return False

    for line in result.stdout.split('\n'):
        # Look for Chrome main process (not helper processes)
        if 'Google Chrome.app/Contents/MacOS/Google Chrome' not in line:
            continue
        # Check for Selenium indicators
        if '--test-type=webdriver' in line:
            return True
        # Check for temp user-data-dir (Selenium uses temp directories)
        if '--user-data-dir=/var/folders/' in line or '--user-data-dir=/tmp/' in line:
            return True

    return False


def extract_live_session_storage(
    origins_filter: Sequence[str] | None = None,
    timeout_seconds: float = 30.0,
) -> AppleScriptExtractionResult:
    """Extract live sessionStorage from running Chrome tabs via AppleScript.

    This is the optimized two-phase approach:
    1. Get all tab URLs (fast, no JS execution)
    2. Filter by origins_filter
    3. Execute JS only on matching tabs (slow part)

    Args:
        origins_filter: Only extract from tabs matching these domains.
                       Example: ["amazon.com", "github.com"]
                       If None, extracts from ALL open tabs (slow).
        timeout_seconds: Maximum time for entire extraction.

    Returns:
        AppleScriptExtractionResult with sessionStorage data or error info.

    Failure Modes:
        - Chrome not running: success=False, error_reason="Chrome not running"
        - Setting disabled: success=False, error_reason="JavaScript from Apple Events disabled"
        - No matching tabs: success=True, session_storage={}, tabs_matched=0
        - Partial success: success=True, tabs_extracted < tabs_matched (some failed)
    """
    if not is_applescript_available():
        return AppleScriptExtractionResult(
            success=False,
            session_storage={},
            source='unavailable',
            error_reason='AppleScript not available (not macOS or Chrome not installed)',
        )

    if not is_chrome_running():
        return AppleScriptExtractionResult(
            success=False,
            session_storage={},
            source='unavailable',
            error_reason='Chrome not running',
        )

    # Fail-fast if Selenium Chrome is running - AppleScript would target wrong browser
    if is_selenium_chrome_running():
        raise SeleniumChromeConflictError(
            'Selenium-controlled Chrome detected. AppleScript cannot reliably distinguish '
            'between Selenium Chrome and your personal Chrome.\n\n'
            'Options:\n'
            '  1. Use browser="chromium" for Selenium to avoid conflicts\n'
            '  2. Close the Selenium browser before extracting from personal Chrome\n'
            '  3. Set live_session_storage=False to skip live extraction (uses disk, may be stale)'
        )

    # Test if JavaScript execution is enabled
    if not _test_javascript_execution():
        return AppleScriptExtractionResult(
            success=False,
            session_storage={},
            source='unavailable',
            error_reason='JavaScript from Apple Events disabled',
        )

    # Phase 1: Get all tab URLs (fast)
    tabs = _get_all_tab_urls()
    if not tabs:
        return AppleScriptExtractionResult(
            success=True,
            session_storage={},
            source='live',
            tabs_matched=0,
            tabs_extracted=0,
        )

    # Phase 2: Filter by origins
    matching_tabs = []
    for window_idx, tab_idx, url in tabs:
        if origins_filter is None:
            matching_tabs.append((window_idx, tab_idx, url))
        else:
            url_domain = _extract_domain_from_url(url)
            if any(_domain_matches(url_domain, pattern) for pattern in origins_filter):
                matching_tabs.append((window_idx, tab_idx, url))

    if not matching_tabs:
        return AppleScriptExtractionResult(
            success=True,
            session_storage={},
            source='live',
            tabs_matched=0,
            tabs_extracted=0,
        )

    # Phase 3: Extract sessionStorage from matching tabs
    session_storage: dict[str, dict[str, str]] = {}
    tabs_extracted = 0

    for window_idx, tab_idx, url in matching_tabs:
        origin = _get_origin_from_url(url)
        if not origin:
            continue

        storage = _extract_session_storage_from_tab(window_idx, tab_idx)
        if storage is not None:
            # Normalize origin (strip trailing slash to match window.location.origin)
            normalized_origin = origin.rstrip('/')
            if normalized_origin in session_storage:
                # Merge with existing (multiple tabs same origin)
                session_storage[normalized_origin].update(storage)
            else:
                session_storage[normalized_origin] = storage
            tabs_extracted += 1

    return AppleScriptExtractionResult(
        success=True,
        session_storage=session_storage,
        source='live',
        tabs_matched=len(matching_tabs),
        tabs_extracted=tabs_extracted,
    )


# =============================================================================
# Private Helper Functions
# =============================================================================


def _test_javascript_execution() -> bool:
    """Test if JavaScript execution is enabled by running a trivial script.

    Returns True if we can execute JavaScript in Chrome tabs.
    Returns False if the setting "Allow JavaScript from Apple Events" is disabled.
    """
    script = """
tell application "Google Chrome"
    if (count of windows) > 0 then
        if (count of tabs of window 1) > 0 then
            execute tab 1 of window 1 javascript "1+1"
            return "ok"
        end if
    end if
    return "no_tabs"
end tell
"""
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # Check for the specific error when setting is disabled
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if 'not authorized' in stderr or 'applescript' in stderr:
            return False
    return result.returncode == 0


def _get_all_tab_urls() -> list[tuple[int, int, str]]:
    """Get URLs of all Chrome tabs: [(window_index, tab_index, url), ...]

    Uses AppleScript to enumerate tabs without executing JavaScript.
    This is fast since it only reads tab properties.
    """
    script = """
set output to ""
tell application "Google Chrome"
    repeat with w from 1 to count of windows
        repeat with t from 1 to count of tabs of window w
            set tabURL to URL of tab t of window w
            set output to output & w & "," & t & "," & tabURL & linefeed
        end repeat
    end repeat
end tell
return output
"""
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return []

    tabs = []
    for line in result.stdout.strip().split('\n'):
        if ',' in line:
            parts = line.split(',', 2)
            if len(parts) == 3:
                try:
                    tabs.append((int(parts[0]), int(parts[1]), parts[2]))
                except ValueError:
                    continue
    return tabs


def _extract_session_storage_from_tab(window: int, tab: int) -> dict[str, str] | None:
    """Extract sessionStorage from a single tab.

    Returns dict of {key: value} or None if extraction failed.
    """
    js_code = 'JSON.stringify(Object.fromEntries(Object.entries(sessionStorage)))'

    script = f'''
tell application "Google Chrome"
    tell window {window}
        execute tab {tab} javascript "{js_code}"
    end tell
end tell
'''
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return {}

    parsed: dict[str, str] = json.loads(output)
    return parsed


def _get_origin_from_url(url: str) -> str | None:
    """Extract origin (scheme://host:port) from URL.

    Returns None for invalid URLs or special URLs (chrome://, about:, etc.)
    """
    if not url:
        return None

    # Skip special URLs
    if url.startswith(('chrome://', 'about:', 'data:', 'blob:', 'file://', 'javascript:')):
        return None

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        return None

    # Include port only if non-standard
    if parsed.port and parsed.port not in (80, 443):
        return f'{parsed.scheme}://{parsed.hostname}:{parsed.port}'
    return f'{parsed.scheme}://{parsed.hostname}'


def _extract_domain_from_url(url: str) -> str:
    """Extract domain from URL for matching.

    Similar to _extract_domain_from_origin in chrome_profile_state_export.py
    """
    if not url:
        return ''

    # Handle full URLs
    if '://' in url:
        # Remove scheme
        url = url.split('://', 1)[1]
        # Remove path
        url = url.split('/', 1)[0]
        # Remove port
        url = url.split(':', 1)[0]
        return url

    return url


def _domain_matches(host: str, pattern: str) -> bool:
    """RFC 6265 domain matching - suffix match with dot boundary.

    Args:
        host: The domain from URL (e.g., "www.amazon.com")
        pattern: The filter pattern (e.g., "amazon.com")

    Returns:
        True if host matches pattern per RFC 6265 rules
    """
    # Canonicalize: lowercase, strip leading/trailing dots
    host = host.lower().strip('.')
    pattern = pattern.lower().strip('.')

    # Exact match
    if host == pattern:
        return True

    # Suffix match with dot boundary: host ends with ".pattern"
    return host.endswith('.' + pattern)
