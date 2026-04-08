from __future__ import annotations

__all__ = [
    'register_tools',
]

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..models import (
    DownloadResourceResult,
    JavaScriptResult,
    ResizeWindowResult,
)
from ..service import BrowserService


def register_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register browser mgmt tools."""

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

    @mcp.tool(
        annotations=ToolAnnotations(
            title='Execute JavaScript',
            readOnlyHint=False,
            idempotentHint=False,
        ),
    )
    async def execute_javascript(code: str, ctx: Context[Any, Any, Any], timeout_ms: int = 30000) -> JavaScriptResult:
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
        return await service.execute_javascript(code=code, timeout_ms=timeout_ms)

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
