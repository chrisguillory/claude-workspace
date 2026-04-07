from __future__ import annotations

__all__ = [
    'register_proxy_tools',
]

from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    ClearProxyResult,
    ConfigureProxyResult,
    SetBlockedURLsResult,
)
from ..service import BrowserService


def register_proxy_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register proxy tools."""

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
        return await service.configure_proxy(host=host, port=port, username=username, password=password)

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
