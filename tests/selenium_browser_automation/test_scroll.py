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
from selenium_browser_automation.scroll import execute_scroll

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
_overflow_specs = [s for s in _all_specs if s['section'] == 'overflow_metadata']

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
        execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
        )

    result = execute_scroll(
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
            execute_scroll(
                headless_driver,
                direction=spec.get('direction'),
                scroll_amount=spec.get('scroll_amount', 3),
                css_selector=spec.get('css_selector'),
            )
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = execute_scroll(
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

    # Setup: scroll to element first (for "scroll twice -> scrolled:false" tests)
    if 'setup_scroll_into_view' in spec:
        execute_scroll(headless_driver, css_selector=spec['setup_scroll_into_view'])

    expect = spec['expect']

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            execute_scroll(headless_driver, css_selector=spec['css_selector'])
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = execute_scroll(headless_driver, css_selector=spec['css_selector'])

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
            execute_scroll(
                headless_driver,
                direction=direction,
                scroll_amount=scroll_amount,
                css_selector=css_selector,
            )
        assert expect['error_contains'] in str(exc_info.value), (
            f'Expected error containing {expect["error_contains"]!r}, got {str(exc_info.value)!r}'
        )
    else:
        # Valid params -- need a page loaded
        headless_driver.get(f'{examples_server}/examples/scroll-test.html')
        headless_driver.execute_script('window.resetScrollPosition()')
        result = execute_scroll(
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

    # Setup: scroll to element first (for "scroll twice -> scrolled:false" tests)
    if 'setup_scroll_into_view' in spec:
        execute_scroll(
            headless_driver,
            css_selector=spec['setup_scroll_into_view'],
            behavior=spec.get('behavior', 'instant'),
        )

    expect = spec['expect']

    result = execute_scroll(
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
        execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
        )

    # Run setup container scroll
    if 'setup_container_scroll' in spec:
        setup = spec['setup_container_scroll']
        execute_scroll(
            headless_driver,
            direction=setup['direction'],
            scroll_amount=setup.get('scroll_amount', 3),
            css_selector=setup.get('css_selector'),
        )

    # Run setup position (e.g., go to bottom before testing already-at-bottom)
    if 'setup_position' in spec:
        execute_scroll(
            headless_driver,
            position=spec['setup_position'],
        )

    if not expect['success']:
        with pytest.raises(ValueError) as exc_info:
            execute_scroll(
                headless_driver,
                position=spec.get('position'),
                direction=spec.get('direction'),
                css_selector=spec.get('css_selector'),
                behavior=spec.get('behavior', 'instant'),
            )
        assert expect['error_contains'] in str(exc_info.value)
        return

    result = execute_scroll(
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
# OVERFLOW METADATA TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _overflow_specs, ids=lambda tc: tc['id'])
def test_overflow_metadata(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test overflow CSS metadata in container scroll responses."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')
    headless_driver.execute_script('window.resetScrollPosition()')

    expect = spec['expect']

    result = execute_scroll(
        headless_driver,
        direction=spec.get('direction'),
        scroll_amount=spec.get('scroll_amount', 3),
        css_selector=spec.get('css_selector'),
    )

    if 'scrolled' in expect:
        assert result['scrolled'] == expect['scrolled'], (
            f'Expected scrolled={expect["scrolled"]}, got {result["scrolled"]}'
        )
    if 'overflow' in expect:
        assert result.get('overflow') == expect['overflow'], (
            f'Expected overflow={expect["overflow"]!r}, got {result.get("overflow")!r}'
        )
