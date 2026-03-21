"""YAML-driven integration tests for scroll tool.

Tests the scroll tool's three modes: viewport scrolling via ActionChains,
container scrolling via JS element.scrollBy(), and scroll-into-view via
JS scrollIntoView(). Each test category exercises a different mode using
JavaScript validation helpers in the test fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium_browser_automation.scripts import SMOOTH_SCROLL_SCRIPT

from tests.selenium_browser_automation.helpers import load_yaml_test_specs

_YAML_PATH = Path(__file__).parent / 'scroll_tests.yaml'
_all_specs = [s for s in load_yaml_test_specs(_YAML_PATH) if s['spec'].get('status') != 'planned']

# Partition by section
_viewport_specs = [s for s in _all_specs if s['section'] == 'viewport_scroll']
_container_specs = [s for s in _all_specs if s['section'] == 'container_scroll']
_scroll_to_specs = [s for s in _all_specs if s['section'] == 'scroll_to_element']
_validation_specs = [s for s in _all_specs if s['section'] == 'parameter_validation']
_smooth_specs = [s for s in _all_specs if s['section'] == 'smooth_scroll']
_position_specs = [s for s in _all_specs if s['section'] == 'position_scroll']

PIXELS_PER_TICK = 100


# ============================================================================
# VIEWPORT SCROLL TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _viewport_specs, ids=lambda tc: tc['id'])
def test_viewport_scroll(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test directional viewport scrolling."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')
    headless_driver.execute_script('window.resetScrollPosition()')

    # Run setup scroll if specified (e.g., scroll down before testing scroll up)
    if 'setup_scroll' in spec:
        setup = spec['setup_scroll']
        _execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
        )

    result = _execute_scroll(
        headless_driver,
        direction=spec.get('direction'),
        scroll_amount=spec.get('scroll_amount', 3),
    )

    expect = spec['expect']
    scroll_y = result.get('scroll_y', headless_driver.execute_script('return window.getViewportScrollY()'))

    if 'viewport_scrollY_gt' in expect:
        assert scroll_y > expect['viewport_scrollY_gt'], (
            f'Expected scrollY > {expect["viewport_scrollY_gt"]}, got {scroll_y}'
        )
    if 'viewport_scrollY_lt' in expect:
        assert scroll_y < expect['viewport_scrollY_lt'], (
            f'Expected scrollY < {expect["viewport_scrollY_lt"]}, got {scroll_y}'
        )
    if 'viewport_scrollY_eq' in expect:
        assert scroll_y == expect['viewport_scrollY_eq'], (
            f'Expected scrollY == {expect["viewport_scrollY_eq"]}, got {scroll_y}'
        )
    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )


# ============================================================================
# CONTAINER SCROLL TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _container_specs, ids=lambda tc: tc['id'])
def test_container_scroll(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test directional scrolling within a container element."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')
    headless_driver.execute_script('window.resetScrollPosition()')

    expect = spec['expect']

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            _execute_scroll(
                headless_driver,
                direction=spec.get('direction'),
                scroll_amount=spec.get('scroll_amount', 3),
                css_selector=spec.get('css_selector'),
            )
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = _execute_scroll(
        headless_driver,
        direction=spec.get('direction'),
        scroll_amount=spec.get('scroll_amount', 3),
        css_selector=spec.get('css_selector'),
    )

    if 'container_scrollTop_gt' in expect:
        assert result['container_scroll_top'] > expect['container_scrollTop_gt'], (
            f'Expected scrollTop > {expect["container_scrollTop_gt"]}, got {result["container_scroll_top"]}'
        )
    if 'container_scrollLeft_gt' in expect:
        assert result['container_scroll_left'] > expect['container_scrollLeft_gt'], (
            f'Expected scrollLeft > {expect["container_scrollLeft_gt"]}, got {result["container_scroll_left"]}'
        )
    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )


# ============================================================================
# SCROLL TO ELEMENT TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _scroll_to_specs, ids=lambda tc: tc['id'])
def test_scroll_to_element(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test scroll-element-into-view behavior."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')
    headless_driver.execute_script('window.resetScrollPosition()')

    # Setup: scroll to element first (for "scroll twice → scrolled:false" tests)
    if 'setup_scroll_into_view' in spec:
        _execute_scroll(headless_driver, css_selector=spec['setup_scroll_into_view'])

    expect = spec['expect']

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            _execute_scroll(headless_driver, css_selector=spec['css_selector'])
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = _execute_scroll(headless_driver, css_selector=spec['css_selector'])

    if 'element_in_viewport' in expect:
        in_viewport = headless_driver.execute_script('return window.isInViewport(arguments[0])', spec['css_selector'])
        assert in_viewport == expect['element_in_viewport'], (
            f'Expected element {spec["css_selector"]} in viewport={expect["element_in_viewport"]}, got {in_viewport}'
        )
    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )
    if 'has_keys' in expect:
        for key in expect['has_keys']:
            assert key in result, f'Expected key {key!r} in result, got keys: {list(result.keys())}'


# ============================================================================
# PARAMETER VALIDATION TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _validation_specs, ids=lambda tc: tc['id'])
def test_scroll_parameter_validation(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test parameter validation.

    Most validation tests don't need the browser, but direction_only_valid
    needs a page to scroll on. We load the fixture for all tests for simplicity.
    """
    spec = test_case['spec']
    expect = spec['expect']

    direction = spec.get('direction')
    scroll_amount = spec.get('scroll_amount', 3)
    css_selector = spec.get('css_selector')

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            _execute_scroll(
                headless_driver,
                direction=direction,
                scroll_amount=scroll_amount,
                css_selector=css_selector,
            )
        assert expect['error_contains'] in str(exc_info.value), (
            f'Expected error containing {expect["error_contains"]!r}, got {str(exc_info.value)!r}'
        )
    else:
        # Valid params — need a page loaded
        headless_driver.get(f'{examples_server}/examples/scroll-test.html')
        headless_driver.execute_script('window.resetScrollPosition()')
        result = _execute_scroll(
            headless_driver,
            direction=direction,
            scroll_amount=scroll_amount,
            css_selector=css_selector,
        )
        assert result is not None


# ============================================================================
# SMOOTH SCROLL TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _smooth_specs, ids=lambda tc: tc['id'])
def test_smooth_scroll(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test smooth scroll behavior across all three modes."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')
    headless_driver.execute_script('window.resetScrollPosition()')

    # Setup: scroll to element first (for "scroll twice → scrolled:false" tests)
    if 'setup_scroll_into_view' in spec:
        _execute_scroll(
            headless_driver,
            css_selector=spec['setup_scroll_into_view'],
            behavior=spec.get('behavior', 'instant'),
        )

    expect = spec['expect']

    result = _execute_scroll(
        headless_driver,
        direction=spec.get('direction'),
        scroll_amount=spec.get('scroll_amount', 3),
        css_selector=spec.get('css_selector'),
        behavior=spec.get('behavior', 'instant'),
    )

    # Viewport assertions
    if 'viewport_scrollY_gt' in expect:
        scroll_y = result.get('scroll_y', 0)
        assert scroll_y > expect['viewport_scrollY_gt'], (
            f'Expected scrollY > {expect["viewport_scrollY_gt"]}, got {scroll_y}'
        )
    if 'viewport_scrollY_eq' in expect:
        scroll_y = result.get('scroll_y', 0)
        assert scroll_y == expect['viewport_scrollY_eq'], (
            f'Expected scrollY == {expect["viewport_scrollY_eq"]}, got {scroll_y}'
        )

    # Container assertions
    if 'container_scrollTop_gt' in expect:
        assert result['container_scroll_top'] > expect['container_scrollTop_gt'], (
            f'Expected scrollTop > {expect["container_scrollTop_gt"]}, got {result["container_scroll_top"]}'
        )

    # Scroll-into-view assertions
    if 'element_in_viewport' in expect:
        in_viewport = headless_driver.execute_script('return window.isInViewport(arguments[0])', spec['css_selector'])
        assert in_viewport == expect['element_in_viewport'], (
            f'Expected in viewport={expect["element_in_viewport"]}, got {in_viewport}'
        )

    # Scrolled boolean
    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )


# ============================================================================
# POSITION SCROLL TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _position_specs, ids=lambda tc: tc['id'])
def test_position_scroll(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test absolute position scrolling (scrollTo) for viewport and containers."""
    spec = test_case['spec']
    expect = spec['expect']

    # Load fixture if specified
    if 'fixture' in spec:
        headless_driver.get(f'{examples_server}/{spec["fixture"]}')
        headless_driver.execute_script('window.resetScrollPosition()')

    # Run setup scroll (e.g., scroll down before testing position='top')
    if 'setup_scroll' in spec:
        setup = spec['setup_scroll']
        _execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
        )

    # Run setup container scroll
    if 'setup_container_scroll' in spec:
        setup = spec['setup_container_scroll']
        _execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
            css_selector=setup.get('css_selector'),
        )

    # Run setup position (e.g., go to bottom before testing already-at-bottom)
    if 'setup_position' in spec:
        _execute_scroll(
            headless_driver,
            position=spec['setup_position'],
        )

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            _execute_scroll(
                headless_driver,
                position=spec.get('position'),
                direction=spec.get('direction'),
                css_selector=spec.get('css_selector'),
                behavior=spec.get('behavior', 'instant'),
            )
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = _execute_scroll(
        headless_driver,
        position=spec.get('position'),
        css_selector=spec.get('css_selector'),
        behavior=spec.get('behavior', 'instant'),
    )

    # Viewport assertions
    if 'viewport_scrollY_gt' in expect:
        assert result.get('scroll_y', 0) > expect['viewport_scrollY_gt'], (
            f'Expected scrollY > {expect["viewport_scrollY_gt"]}, got {result.get("scroll_y")}'
        )
    if 'viewport_scrollY_eq' in expect:
        assert result.get('scroll_y', 0) == expect['viewport_scrollY_eq'], (
            f'Expected scrollY == {expect["viewport_scrollY_eq"]}, got {result.get("scroll_y")}'
        )

    # Container assertions
    if 'container_scrollTop_gt' in expect:
        assert result['container_scroll_top'] > expect['container_scrollTop_gt'], (
            f'Expected scrollTop > {expect["container_scrollTop_gt"]}, got {result["container_scroll_top"]}'
        )
    if 'container_scrollTop_eq' in expect:
        assert result['container_scroll_top'] == expect['container_scrollTop_eq'], (
            f'Expected scrollTop == {expect["container_scrollTop_eq"]}, got {result["container_scroll_top"]}'
        )
    if 'container_scrollLeft_gt' in expect:
        assert result['container_scroll_left'] > expect['container_scrollLeft_gt'], (
            f'Expected scrollLeft > {expect["container_scrollLeft_gt"]}, got {result["container_scroll_left"]}'
        )

    # Scrolled boolean
    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )


# ============================================================================
# SCROLL EXECUTION HELPER
# ============================================================================


def _execute_scroll(
    driver: webdriver.Chrome,
    direction: str | None = None,
    scroll_amount: int = 3,
    css_selector: str | None = None,
    behavior: str = 'instant',
    position: str | None = None,
) -> dict[str, Any]:
    """Replicate scroll tool logic for testing.

    Five modes:
    1. Scroll into view (css_selector only)
    2. Viewport scroll (direction only)
    3. Container scroll (direction + css_selector)
    4. Viewport scrollTo (position only)
    5. Container scrollTo (position + css_selector)
    """
    # Validation
    if position is not None and direction is not None:
        raise ValueError("Cannot specify both 'position' and 'direction'.")
    if direction is None and css_selector is None and position is None:
        raise ValueError('Either direction, css_selector, or position must be provided.')
    if direction is not None and (scroll_amount < 1 or scroll_amount > 20):
        raise ValueError('scroll_amount must be between 1 and 20')

    # Position scrolling (scrollTo)
    if position is not None:
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
            try:
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
                )
            except TimeoutException:
                raise ValueError(f"Failed to find container '{css_selector}'") from None

            scroll_target = pos_map_container[position]
            if behavior == 'smooth':
                result = driver.execute_script(
                    SMOOTH_SCROLL_SCRIPT
                    + f"""
                    var el = arguments[0];
                    var beforeTop = Math.round(el.scrollTop), beforeLeft = Math.round(el.scrollLeft);
                    return smoothScroll({{
                        eventTarget: el,
                        scrollAction: function() {{ el.scrollTo({{{scroll_target}, behavior: 'smooth'}}); }},
                        hasMoved: function() {{
                            return Math.round(el.scrollTop) !== beforeTop
                                || Math.round(el.scrollLeft) !== beforeLeft;
                        }},
                        measure: function() {{
                            return {{
                                scrollTop: Math.round(el.scrollTop), scrollLeft: Math.round(el.scrollLeft),
                                scrollHeight: el.scrollHeight, clientHeight: el.clientHeight,
                                scrolled: (Math.round(el.scrollTop) !== beforeTop
                                        || Math.round(el.scrollLeft) !== beforeLeft)
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
                        scrollTop: Math.round(el.scrollTop), scrollLeft: Math.round(el.scrollLeft),
                        scrollHeight: el.scrollHeight, clientHeight: el.clientHeight,
                        scrolled: (Math.round(el.scrollTop) !== beforeTop
                                || Math.round(el.scrollLeft) !== beforeLeft)
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
            }
        else:
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
                                scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
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
                before_y = driver.execute_script('return Math.round(window.scrollY);')
                before_x = driver.execute_script('return Math.round(window.scrollX);')
                driver.execute_script(f'window.scrollTo({{{scroll_target}, behavior: "instant"}});')
                after_info = driver.execute_script("""
                    return {
                        scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
                        pageHeight: Math.max(
                            document.body.scrollHeight, document.documentElement.scrollHeight,
                            document.body.offsetHeight, document.documentElement.offsetHeight
                        ),
                        viewportHeight: window.innerHeight
                    };
                """)
                result = {
                    **after_info,
                    'scrolled': after_info['scrollX'] != before_x or after_info['scrollY'] != before_y,
                }
            return {
                'mode': 'viewport_scroll_to',
                'position': position,
                'scroll_x': result['scrollX'],
                'scroll_y': result['scrollY'],
                'page_height': result['pageHeight'],
                'viewport_height': result['viewportHeight'],
                'scrolled': result['scrolled'],
            }

    # Mode 1: Scroll into view
    if css_selector is not None and direction is None:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
            )
        except TimeoutException:
            raise ValueError(f"Failed to find element '{css_selector}'") from None

        if behavior == 'smooth':
            after_info = driver.execute_script(
                SMOOTH_SCROLL_SCRIPT
                + """
                var target = arguments[0];
                var beforeY = Math.round(window.scrollY);
                return smoothScroll({
                    eventTarget: window,
                    scrollAction: function() {
                        target.scrollIntoView({block: 'center', behavior: 'smooth'});
                    },
                    hasMoved: function() { return Math.round(window.scrollY) !== beforeY; },
                    measure: function() {
                        return {
                            scrollX: Math.round(window.scrollX),
                            scrollY: Math.round(window.scrollY),
                            pageHeight: Math.max(
                                document.body.scrollHeight, document.documentElement.scrollHeight,
                                document.body.offsetHeight, document.documentElement.offsetHeight
                            ),
                            viewportHeight: window.innerHeight,
                            scrolled: Math.round(window.scrollY) !== beforeY
                        };
                    }
                });
                """,
                element,
            )
        else:
            before_y = driver.execute_script('return Math.round(window.scrollY);')
            orig = driver.execute_script(
                'var orig = document.documentElement.style.scrollBehavior;'
                'document.documentElement.style.scrollBehavior = "auto";'
                'return orig;'
            )
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});",
                element,
            )
            driver.execute_script('document.documentElement.style.scrollBehavior = arguments[0];', orig or '')
            after_info = driver.execute_script("""
                return {
                    scrollX: Math.round(window.scrollX),
                    scrollY: Math.round(window.scrollY),
                    pageHeight: Math.max(
                        document.body.scrollHeight, document.documentElement.scrollHeight,
                        document.body.offsetHeight, document.documentElement.offsetHeight
                    ),
                    viewportHeight: window.innerHeight,
                    scrolled: false
                };
            """)
            after_info['scrolled'] = after_info['scrollY'] != before_y

        return {
            'mode': 'scroll_to_element',
            'selector': css_selector,
            'scroll_x': after_info['scrollX'],
            'scroll_y': after_info['scrollY'],
            'page_height': after_info['pageHeight'],
            'viewport_height': after_info['viewportHeight'],
            'scrolled': after_info['scrolled'],
        }

    # Calculate deltas for directional scroll
    assert direction is not None  # tests only — guaranteed by validation above
    delta_map = {
        'up': (0, -PIXELS_PER_TICK * scroll_amount),
        'down': (0, PIXELS_PER_TICK * scroll_amount),
        'left': (-PIXELS_PER_TICK * scroll_amount, 0),
        'right': (PIXELS_PER_TICK * scroll_amount, 0),
    }
    delta_x, delta_y = delta_map[direction]

    # Mode 3: Container scroll
    if css_selector is not None:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
            )
        except TimeoutException:
            raise ValueError(f"Failed to find element '{css_selector}'") from None

        if behavior == 'smooth':
            result = driver.execute_script(
                SMOOTH_SCROLL_SCRIPT
                + """
                var el = arguments[0], dy = arguments[1], dx = arguments[2];
                var beforeTop = Math.round(el.scrollTop), beforeLeft = Math.round(el.scrollLeft);
                return smoothScroll({
                    eventTarget: el,
                    scrollAction: function() { el.scrollBy({top: dy, left: dx, behavior: 'smooth'}); },
                    hasMoved: function() {
                        return Math.round(el.scrollTop) !== beforeTop || Math.round(el.scrollLeft) !== beforeLeft;
                    },
                    measure: function() {
                        return {
                            scrollTop: Math.round(el.scrollTop), scrollLeft: Math.round(el.scrollLeft),
                            scrollHeight: el.scrollHeight, clientHeight: el.clientHeight,
                            scrolled: (Math.round(el.scrollTop) !== beforeTop || Math.round(el.scrollLeft) !== beforeLeft)
                        };
                    }
                });
                """,
                element,
                delta_y,
                delta_x,
            )
        else:
            before = driver.execute_script(
                'return { top: arguments[0].scrollTop, left: arguments[0].scrollLeft };',
                element,
            )
            driver.execute_script(
                'arguments[0].scrollBy({top: arguments[1], left: arguments[2], behavior: "instant"});',
                element,
                delta_y,
                delta_x,
            )
            after = driver.execute_script(
                """
                return {
                    scrollTop: Math.round(arguments[0].scrollTop),
                    scrollLeft: Math.round(arguments[0].scrollLeft),
                    scrollHeight: arguments[0].scrollHeight,
                    clientHeight: arguments[0].clientHeight
                };
            """,
                element,
            )
            result = {
                **after,
                'scrolled': (after['scrollTop'] != before['top'] or after['scrollLeft'] != before['left']),
            }

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
        }

    # Mode 2: Viewport scroll
    if behavior == 'smooth':
        after_info = driver.execute_script(
            SMOOTH_SCROLL_SCRIPT
            + """
            var dy = arguments[0], dx = arguments[1];
            var beforeX = Math.round(window.scrollX), beforeY = Math.round(window.scrollY);
            return smoothScroll({
                eventTarget: window,
                scrollAction: function() { window.scrollBy({top: dy, left: dx, behavior: 'smooth'}); },
                hasMoved: function() {
                    return Math.round(window.scrollX) !== beforeX || Math.round(window.scrollY) !== beforeY;
                },
                measure: function() {
                    return {
                        scrollX: Math.round(window.scrollX), scrollY: Math.round(window.scrollY),
                        pageHeight: Math.max(
                            document.body.scrollHeight, document.documentElement.scrollHeight,
                            document.body.offsetHeight, document.documentElement.offsetHeight
                        ),
                        viewportHeight: window.innerHeight,
                        scrolled: (Math.round(window.scrollX) !== beforeX || Math.round(window.scrollY) !== beforeY)
                    };
                }
            });
            """,
            delta_y,
            delta_x,
        )
    else:
        before_y = driver.execute_script('return Math.round(window.scrollY);')
        before_x = driver.execute_script('return Math.round(window.scrollX);')
        driver.execute_script(
            'window.scrollBy({top: arguments[0], left: arguments[1], behavior: "instant"});',
            delta_y,
            delta_x,
        )
        after_info = driver.execute_script("""
            return {
                scrollX: Math.round(window.scrollX),
                scrollY: Math.round(window.scrollY),
                pageHeight: Math.max(
                    document.body.scrollHeight, document.documentElement.scrollHeight,
                    document.body.offsetHeight, document.documentElement.offsetHeight
                ),
                viewportHeight: window.innerHeight
            };
        """)
        after_info['scrolled'] = after_info['scrollX'] != before_x or after_info['scrollY'] != before_y

    return {
        'mode': 'viewport_scroll',
        'direction': direction,
        'pixels': abs(delta_x) + abs(delta_y),
        'scroll_x': after_info['scrollX'],
        'scroll_y': after_info['scrollY'],
        'page_height': after_info['pageHeight'],
        'viewport_height': after_info['viewportHeight'],
        'scrolled': after_info['scrolled'],
    }
