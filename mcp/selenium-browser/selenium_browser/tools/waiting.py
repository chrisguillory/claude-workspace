from __future__ import annotations

__all__ = [
    'register_tools',
]

from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    SleepResult,
    WaitForSelectorResult,
)
from ..service import BrowserService


def register_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register waiting tools."""

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
        return await service.sleep(duration_ms=duration_ms, reason=reason)
