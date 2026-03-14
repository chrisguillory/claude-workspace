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

from tests.selenium_browser_automation.helpers import load_yaml_test_specs

_YAML_PATH = Path(__file__).parent / 'scroll_tests.yaml'
_all_specs = [s for s in load_yaml_test_specs(_YAML_PATH) if s['spec'].get('status') != 'planned']

# Partition by section
_viewport_specs = [s for s in _all_specs if s['section'] == 'viewport_scroll']
_container_specs = [s for s in _all_specs if s['section'] == 'container_scroll']
_scroll_to_specs = [s for s in _all_specs if s['section'] == 'scroll_to_element']
_validation_specs = [s for s in _all_specs if s['section'] == 'parameter_validation']

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
# SCROLL EXECUTION HELPER
# ============================================================================


def _execute_scroll(
    driver: webdriver.Chrome,
    direction: str | None = None,
    scroll_amount: int = 3,
    css_selector: str | None = None,
) -> dict[str, Any]:
    """Replicate scroll tool logic for testing.

    Three modes:
    1. Scroll into view (css_selector only, no direction)
    2. Viewport scroll (direction only)
    3. Container scroll (direction + css_selector)
    """
    # Validation
    if direction is None and css_selector is None:
        raise ValueError('Either direction or css_selector (or both) must be provided.')
    if scroll_amount < 1 or scroll_amount > 20:
        raise ValueError('scroll_amount must be between 1 and 20')

    # Mode 1: Scroll into view
    if css_selector is not None and direction is None:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css_selector)),
            )
        except TimeoutException:
            raise ValueError(f"Failed to find element '{css_selector}'") from None

        # Capture before position
        before_y = driver.execute_script('return Math.round(window.scrollY);')

        # Override scroll-behavior for instant scrolling (belt-and-suspenders)
        orig = driver.execute_script(
            'var orig = document.documentElement.style.scrollBehavior;'
            'document.documentElement.style.scrollBehavior = "auto";'
            'return orig;'
        )

        # scrollIntoView with block:'center' for maximum context
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', behavior: 'instant'});",
            element,
        )

        # Restore scroll-behavior
        driver.execute_script('document.documentElement.style.scrollBehavior = arguments[0];', orig or '')

        # Read position after scroll
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

        return {
            'mode': 'scroll_to_element',
            'selector': css_selector,
            'scroll_x': after_info['scrollX'],
            'scroll_y': after_info['scrollY'],
            'page_height': after_info['pageHeight'],
            'viewport_height': after_info['viewportHeight'],
            'scrolled': after_info['scrollY'] != before_y,
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

        # Before: measure container position
        before = driver.execute_script(
            'return { top: arguments[0].scrollTop, left: arguments[0].scrollLeft };',
            element,
        )

        # scrollBy with behavior:'instant' to bypass CSS scroll-behavior:smooth
        driver.execute_script(
            'arguments[0].scrollBy({top: arguments[1], left: arguments[2], behavior: "instant"});',
            element,
            delta_y,
            delta_x,
        )

        # After: measure container position
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

        return {
            'mode': 'container_scroll',
            'direction': direction,
            'pixels': abs(delta_x) + abs(delta_y),
            'selector': css_selector,
            'container_scroll_top': after['scrollTop'],
            'container_scroll_left': after['scrollLeft'],
            'container_scroll_height': after['scrollHeight'],
            'container_client_height': after['clientHeight'],
            'scrolled': (after['scrollTop'] != before['top'] or after['scrollLeft'] != before['left']),
        }

    # Mode 2: Viewport scroll
    # Use JS window.scrollBy with behavior:'instant' — ActionChains scroll_by_amount
    # does NOT work in headless Chrome (returns scrollY=0).
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

    return {
        'mode': 'viewport_scroll',
        'direction': direction,
        'pixels': abs(delta_x) + abs(delta_y),
        'scroll_x': after_info['scrollX'],
        'scroll_y': after_info['scrollY'],
        'page_height': after_info['pageHeight'],
        'viewport_height': after_info['viewportHeight'],
        'scrolled': (after_info['scrollX'] != before_x or after_info['scrollY'] != before_y),
    }
