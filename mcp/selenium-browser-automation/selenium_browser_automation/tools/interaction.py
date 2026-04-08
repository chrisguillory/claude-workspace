from __future__ import annotations

__all__ = [
    'register_tools',
]

from typing import Any, Literal

from cc_lib.types import JsonObject
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from ..service import BrowserService


def register_tools(service: BrowserService, mcp: FastMCP) -> None:
    """Register interaction tools."""

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
