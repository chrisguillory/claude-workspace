from __future__ import annotations

__all__ = [
    'register_tools',
]

from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    ConsoleLogsResult,
    CoreWebVitals,
    HARExportResult,
    NetworkCapture,
)
from ..service import BrowserService


def register_tools(service: BrowserService, mcp: FastMCP) -> None:
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
        annotations=ToolAnnotations(
            title='Get Resource Timings',
            readOnlyHint=False,  # clear_resource_timing_buffer=True clears the buffer
            idempotentHint=False,
        ),
    )
    async def get_resource_timings(
        ctx: Context[Any, Any, Any],
        clear_resource_timing_buffer: bool = False,
        min_duration_ms: int = 0,
    ) -> NetworkCapture:
        """Get resource timing data from the browser's Performance API.

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
            with many resources, use clear_resource_timing_buffer=True to prevent data loss.
        """
        return await service.get_resource_timings(
            clear_resource_timing_buffer=clear_resource_timing_buffer, min_duration_ms=min_duration_ms
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Export HAR',
            readOnlyHint=False,  # Writes file to disk
            idempotentHint=True,
        ),
    )
    async def export_har(
        ctx: Context[Any, Any, Any],
        filename: str,
        include_response_bodies: bool = False,
        max_body_size_mb: int = 10,
    ) -> HARExportResult:
        """Export captured network traffic to HAR 1.2 file.

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
            4. Read the HAR file or import into Chrome DevTools
        """
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
