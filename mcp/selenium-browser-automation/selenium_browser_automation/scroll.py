"""Synchronous scroll logic for Selenium WebDriver.

Extracted from the MCP tool for direct use in tests and the async wrapper.
All modes: viewport scroll, container scroll, scroll-into-view,
viewport scrollTo, and container scrollTo — with instant and smooth behavior.
"""

from __future__ import annotations

from typing import Any, Literal

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .scripts import SMOOTH_SCROLL_SCRIPT
from .validators import validate_css_selector

__all__ = ['PIXELS_PER_TICK', 'execute_scroll', 'validate_css_selector']

PIXELS_PER_TICK = 100


def execute_scroll(
    driver: webdriver.Chrome,
    *,
    direction: Literal['up', 'down', 'left', 'right'] | None = None,
    scroll_amount: int = 3,
    css_selector: str | None = None,
    behavior: Literal['instant', 'smooth'] = 'instant',
    position: Literal['top', 'bottom', 'left', 'right'] | None = None,
) -> dict[str, Any]:
    """Execute a scroll operation synchronously.

    Five modes:
    1. Viewport scroll (direction only)
    2. Container scroll (direction + css_selector)
    3. Scroll into view (css_selector only)
    4. Viewport scrollTo (position only)
    5. Container scrollTo (position + css_selector)

    Raises ValueError for invalid parameters or missing elements.
    """
    # ── Parameter validation ──

    if position is not None and direction is not None:
        raise ValueError(
            "Cannot specify both 'position' and 'direction'.\n"
            "Use position='top'/'bottom'/'left'/'right' to scroll to an edge, "
            "or direction='up'/'down'/'left'/'right' to scroll by an amount."
        )

    if direction is None and css_selector is None and position is None:
        raise ValueError(
            'Either direction, css_selector, or position must be provided.\n'
            "Use direction='down' for viewport scrolling, "
            "css_selector='#element' to scroll into view, "
            "or position='bottom' to jump to an edge."
        )

    if direction is not None and (scroll_amount < 1 or scroll_amount > 20):
        raise ValueError('scroll_amount must be between 1 and 20')

    # ── Mode 4/5: Absolute position scroll (position parameter) ──

    if position is not None:
        return _scroll_to_position(driver, position=position, css_selector=css_selector, behavior=behavior)

    # ── Mode 3: Scroll element into view (css_selector only, no direction) ──

    if css_selector is not None and direction is None:
        return _scroll_into_view(driver, css_selector=css_selector, behavior=behavior)

    # ── Directional scroll (modes 1 and 2) ──

    delta_map = {
        'up': (0, -PIXELS_PER_TICK * scroll_amount),
        'down': (0, PIXELS_PER_TICK * scroll_amount),
        'left': (-PIXELS_PER_TICK * scroll_amount, 0),
        'right': (PIXELS_PER_TICK * scroll_amount, 0),
    }
    delta_x, delta_y = delta_map[direction]  # type: ignore[index]

    if css_selector is not None:
        return _scroll_container(
            driver,
            direction=direction,
            css_selector=css_selector,
            delta_x=delta_x,
            delta_y=delta_y,
            behavior=behavior,
        )

    return _scroll_viewport(
        driver,
        direction=direction,
        delta_x=delta_x,
        delta_y=delta_y,
        behavior=behavior,
    )


# ── Private helpers ──


def _find_element(driver: webdriver.Chrome, css_selector: str, context: str) -> Any:
    """Find element by CSS selector with wait. Raises ValueError if not found."""
    validate_css_selector(css_selector)
    try:
        return WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
        )
    except TimeoutException:
        raise ValueError(
            f"Failed to find {context} '{css_selector}'.\nUse get_aria_snapshot('body') to discover valid selectors."
        ) from None


def _scroll_to_position(
    driver: webdriver.Chrome,
    *,
    position: str,
    css_selector: str | None,
    behavior: str,
) -> dict[str, Any]:
    """Mode 4/5: Absolute position scroll (scrollTo)."""
    pos_map_viewport = {
        'top': 'top: 0, left: 0',
        'bottom': (
            'top: Math.max(document.body.scrollHeight, '
            'document.documentElement.scrollHeight, '
            'document.body.offsetHeight, document.documentElement.offsetHeight), left: 0'
        ),
        'left': 'top: window.scrollY, left: 0',
        'right': (
            'top: window.scrollY, left: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth)'
        ),
    }
    pos_map_container = {
        'top': 'top: 0, left: el.scrollLeft',
        'bottom': 'top: el.scrollHeight, left: el.scrollLeft',
        'left': 'top: el.scrollTop, left: 0',
        'right': 'top: el.scrollTop, left: el.scrollWidth',
    }

    if css_selector is not None:
        # Mode 5: Container scrollTo
        element = _find_element(driver, css_selector, 'container')
        scroll_target = pos_map_container[position]
        overflow_axis = 'overflowY' if position in ('top', 'bottom') else 'overflowX'

        if behavior == 'smooth':
            result = driver.execute_script(
                SMOOTH_SCROLL_SCRIPT
                + f"""
                var el = arguments[0];
                var beforeTop = Math.round(el.scrollTop), beforeLeft = Math.round(el.scrollLeft);
                var overflow = window.getComputedStyle(el).{overflow_axis};
                return smoothScroll({{
                    eventTarget: el,
                    scrollAction: function() {{ el.scrollTo({{{scroll_target}, behavior: 'smooth'}}); }},
                    hasMoved: function() {{
                        return Math.round(el.scrollTop) !== beforeTop
                            || Math.round(el.scrollLeft) !== beforeLeft;
                    }},
                    measure: function() {{
                        return {{
                            scrollTop: Math.round(el.scrollTop),
                            scrollLeft: Math.round(el.scrollLeft),
                            scrollHeight: el.scrollHeight,
                            clientHeight: el.clientHeight,
                            scrolled: (Math.round(el.scrollTop) !== beforeTop
                                    || Math.round(el.scrollLeft) !== beforeLeft),
                            overflow: overflow
                        }};
                    }}
                }});
                """,
                element,
            )
        else:
            result = driver.execute_script(
                f"""
                var el = arguments[0];
                var beforeTop = Math.round(el.scrollTop), beforeLeft = Math.round(el.scrollLeft);
                el.scrollTo({{{scroll_target}, behavior: 'instant'}});
                return {{
                    scrollTop: Math.round(el.scrollTop),
                    scrollLeft: Math.round(el.scrollLeft),
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                    scrolled: (Math.round(el.scrollTop) !== beforeTop
                            || Math.round(el.scrollLeft) !== beforeLeft),
                    overflow: window.getComputedStyle(el).{overflow_axis}
                }};
                """,
                element,
            )

        return {
            'mode': 'container_scroll_to',
            'position': position,
            'selector': css_selector,
            'container_scroll_top': result['scrollTop'],
            'container_scroll_left': result['scrollLeft'],
            'container_scroll_height': result['scrollHeight'],
            'container_client_height': result['clientHeight'],
            'scrolled': result['scrolled'],
            'overflow': result['overflow'],
        }

    # Mode 4: Viewport scrollTo
    scroll_target = pos_map_viewport[position]

    if behavior == 'smooth':
        result = driver.execute_script(
            SMOOTH_SCROLL_SCRIPT
            + f"""
            var beforeX = Math.round(window.scrollX), beforeY = Math.round(window.scrollY);
            return smoothScroll({{
                eventTarget: window,
                scrollAction: function() {{ window.scrollTo({{{scroll_target}, behavior: 'smooth'}}); }},
                hasMoved: function() {{
                    return Math.round(window.scrollX) !== beforeX
                        || Math.round(window.scrollY) !== beforeY;
                }},
                measure: function() {{
                    return {{
                        scrollX: Math.round(window.scrollX),
                        scrollY: Math.round(window.scrollY),
                        pageHeight: Math.max(
                            document.body.scrollHeight, document.documentElement.scrollHeight,
                            document.body.offsetHeight, document.documentElement.offsetHeight
                        ),
                        viewportHeight: window.innerHeight,
                        scrolled: (Math.round(window.scrollX) !== beforeX
                                || Math.round(window.scrollY) !== beforeY)
                    }};
                }}
            }});
            """,
        )
    else:
        result = driver.execute_script(
            f"""
            var beforeX = Math.round(window.scrollX), beforeY = Math.round(window.scrollY);
            window.scrollTo({{{scroll_target}, behavior: 'instant'}});
            return {{
                scrollX: Math.round(window.scrollX),
                scrollY: Math.round(window.scrollY),
                pageHeight: Math.max(
                    document.body.scrollHeight, document.documentElement.scrollHeight,
                    document.body.offsetHeight, document.documentElement.offsetHeight
                ),
                viewportHeight: window.innerHeight,
                scrolled: (Math.round(window.scrollX) !== beforeX
                        || Math.round(window.scrollY) !== beforeY)
            }};
            """,
        )

    return {
        'mode': 'viewport_scroll_to',
        'position': position,
        'scroll_x': result['scrollX'],
        'scroll_y': result['scrollY'],
        'page_height': result['pageHeight'],
        'viewport_height': result['viewportHeight'],
        'scrolled': result['scrolled'],
    }


def _scroll_into_view(
    driver: webdriver.Chrome,
    *,
    css_selector: str,
    behavior: str,
) -> dict[str, Any]:
    """Mode 3: Scroll element into view."""
    element = _find_element(driver, css_selector, 'element')

    if behavior == 'smooth':
        after_info = driver.execute_script(
            SMOOTH_SCROLL_SCRIPT
            + """
            var target = arguments[0];
            var beforeX = Math.round(window.scrollX), beforeY = Math.round(window.scrollY);
            return smoothScroll({
                eventTarget: window,
                scrollAction: function() {
                    target.scrollIntoView({block: 'center', behavior: 'smooth'});
                },
                hasMoved: function() {
                    return Math.round(window.scrollX) !== beforeX
                        || Math.round(window.scrollY) !== beforeY;
                },
                measure: function() {
                    return {
                        scrollX: Math.round(window.scrollX),
                        scrollY: Math.round(window.scrollY),
                        pageHeight: Math.max(
                            document.body.scrollHeight, document.documentElement.scrollHeight,
                            document.body.offsetHeight, document.documentElement.offsetHeight
                        ),
                        viewportHeight: window.innerHeight,
                        scrolled: (Math.round(window.scrollX) !== beforeX
                                || Math.round(window.scrollY) !== beforeY)
                    };
                }
            });
            """,
            element,
        )
        scrolled = after_info['scrolled']
    else:
        before_y = driver.execute_script('return Math.round(window.scrollY);')
        driver.execute_script(
            """
            var orig = document.documentElement.style.scrollBehavior;
            document.documentElement.style.scrollBehavior = 'auto';
            arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});
            document.documentElement.style.scrollBehavior = orig || '';
            """,
            element,
        )
        after_info = driver.execute_script(
            """
            return {
                scrollX: Math.round(window.scrollX),
                scrollY: Math.round(window.scrollY),
                pageHeight: Math.max(
                    document.body.scrollHeight, document.documentElement.scrollHeight,
                    document.body.offsetHeight, document.documentElement.offsetHeight
                ),
                viewportHeight: window.innerHeight
            };
            """,
        )
        scrolled = after_info['scrollY'] != before_y

    return {
        'mode': 'scroll_to_element',
        'selector': css_selector,
        'scroll_x': after_info['scrollX'],
        'scroll_y': after_info['scrollY'],
        'page_height': after_info['pageHeight'],
        'viewport_height': after_info['viewportHeight'],
        'scrolled': scrolled,
    }


def _scroll_container(
    driver: webdriver.Chrome,
    *,
    direction: str | None,
    css_selector: str,
    delta_x: int,
    delta_y: int,
    behavior: str,
) -> dict[str, Any]:
    """Mode 2: Container scroll (direction + css_selector)."""
    element = _find_element(driver, css_selector, 'scrollable container')

    if behavior == 'smooth':
        result = driver.execute_script(
            SMOOTH_SCROLL_SCRIPT
            + """
            var el = arguments[0], dy = arguments[1], dx = arguments[2];
            var beforeTop = Math.round(el.scrollTop), beforeLeft = Math.round(el.scrollLeft);
            var overflow = window.getComputedStyle(el)[dy !== 0 ? 'overflowY' : 'overflowX'];
            return smoothScroll({
                eventTarget: el,
                scrollAction: function() {
                    el.scrollBy({top: dy, left: dx, behavior: 'smooth'});
                },
                hasMoved: function() {
                    return Math.round(el.scrollTop) !== beforeTop
                        || Math.round(el.scrollLeft) !== beforeLeft;
                },
                measure: function() {
                    return {
                        scrollTop: Math.round(el.scrollTop),
                        scrollLeft: Math.round(el.scrollLeft),
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight,
                        scrolled: (Math.round(el.scrollTop) !== beforeTop
                                || Math.round(el.scrollLeft) !== beforeLeft),
                        overflow: overflow
                    };
                }
            });
            """,
            element,
            delta_y,
            delta_x,
        )
    else:
        result = driver.execute_script(
            """
            var el = arguments[0], dy = arguments[1], dx = arguments[2];
            var beforeTop = el.scrollTop, beforeLeft = el.scrollLeft;
            el.scrollBy({top: dy, left: dx, behavior: 'instant'});
            return {
                scrollTop: Math.round(el.scrollTop),
                scrollLeft: Math.round(el.scrollLeft),
                scrollHeight: el.scrollHeight,
                clientHeight: el.clientHeight,
                scrolled: (Math.round(el.scrollTop) !== Math.round(beforeTop)
                        || Math.round(el.scrollLeft) !== Math.round(beforeLeft)),
                overflow: window.getComputedStyle(el)[dy !== 0 ? 'overflowY' : 'overflowX']
            };
            """,
            element,
            delta_y,
            delta_x,
        )

    return {
        'mode': 'container_scroll',
        'direction': direction,
        'pixels': abs(delta_x) + abs(delta_y),
        'selector': css_selector,
        'container_scroll_top': result['scrollTop'],
        'container_scroll_left': result['scrollLeft'],
        'container_scroll_height': result['scrollHeight'],
        'container_client_height': result['clientHeight'],
        'scrolled': result['scrolled'],
        'overflow': result['overflow'],
    }


def _scroll_viewport(
    driver: webdriver.Chrome,
    *,
    direction: str | None,
    delta_x: int,
    delta_y: int,
    behavior: str,
) -> dict[str, Any]:
    """Mode 1: Viewport scroll (direction only)."""
    if behavior == 'smooth':
        result = driver.execute_script(
            SMOOTH_SCROLL_SCRIPT
            + """
            var dy = arguments[0], dx = arguments[1];
            var beforeX = Math.round(window.scrollX), beforeY = Math.round(window.scrollY);
            return smoothScroll({
                eventTarget: window,
                scrollAction: function() {
                    window.scrollBy({top: dy, left: dx, behavior: 'smooth'});
                },
                hasMoved: function() {
                    return Math.round(window.scrollX) !== beforeX
                        || Math.round(window.scrollY) !== beforeY;
                },
                measure: function() {
                    return {
                        scrollX: Math.round(window.scrollX),
                        scrollY: Math.round(window.scrollY),
                        pageHeight: Math.max(
                            document.body.scrollHeight, document.documentElement.scrollHeight,
                            document.body.offsetHeight, document.documentElement.offsetHeight
                        ),
                        viewportHeight: window.innerHeight,
                        scrolled: (Math.round(window.scrollX) !== beforeX
                                || Math.round(window.scrollY) !== beforeY)
                    };
                }
            });
            """,
            delta_y,
            delta_x,
        )
    else:
        result = driver.execute_script(
            """
            var dy = arguments[0], dx = arguments[1];
            var beforeX = window.scrollX, beforeY = window.scrollY;
            window.scrollBy({top: dy, left: dx, behavior: 'instant'});
            return {
                scrollX: Math.round(window.scrollX),
                scrollY: Math.round(window.scrollY),
                pageHeight: Math.max(
                    document.body.scrollHeight, document.documentElement.scrollHeight,
                    document.body.offsetHeight, document.documentElement.offsetHeight
                ),
                viewportHeight: window.innerHeight,
                scrolled: (Math.round(window.scrollX) !== Math.round(beforeX)
                        || Math.round(window.scrollY) !== Math.round(beforeY))
            };
            """,
            delta_y,
            delta_x,
        )

    return {
        'mode': 'viewport_scroll',
        'direction': direction,
        'pixels': abs(delta_x) + abs(delta_y),
        'scroll_x': result['scrollX'],
        'scroll_y': result['scrollY'],
        'page_height': result['pageHeight'],
        'viewport_height': result['viewportHeight'],
        'scrolled': result['scrolled'],
    }
