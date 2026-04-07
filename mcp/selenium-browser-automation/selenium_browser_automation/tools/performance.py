from __future__ import annotations

__all__ = [
    'register_performance_tools',
]

import asyncio
from typing import Any, Literal

from cc_lib.types import JsonObject
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from selenium import webdriver

from ..models import (
    ConsoleLogsResult,
    CoreWebVitals,
    HARExportResult,
    NetworkCapture,
)
from ..service import BrowserService


def register_performance_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register performance tools."""

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
        return await service.export_har(
            filename=filename, include_response_bodies=include_response_bodies, max_body_size_mb=max_body_size_mb
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
