"""YAML-driven integration tests for hover tool actionability checks.

Tests the hover tool's pre-hover pipeline: duration validation, visibility
detection, stability analysis, and occlusion detection. Each test category
exercises a different stage of the pipeline using the same JavaScript checks
the MCP server executes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium_browser_automation.hover import check_hover_actionability, validate_hover_duration
from selenium_browser_automation.scripts import HOVER_STABILITY_SCRIPT

from tests.selenium_browser_automation.helpers import load_yaml_test_specs

_YAML_PATH = Path(__file__).parent / 'hover_tests.yaml'
_all_specs = [s for s in load_yaml_test_specs(_YAML_PATH) if s['spec'].get('status') != 'planned']

# Partition by section
_stability_specs = [s for s in _all_specs if s['section'] == 'stability']
_visibility_specs = [s for s in _all_specs if s['section'] == 'visibility']
_occlusion_specs = [s for s in _all_specs if s['section'] == 'occlusion']
_duration_specs = [s for s in _all_specs if s['section'] == 'duration_validation']


# ============================================================================
# STABILITY TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _stability_specs, ids=lambda tc: tc['id'])
def test_hover_stability(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test multi-signal stability detection for animated/static elements."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')

    # Handle setup actions (e.g., start finite animation)
    if 'setup_selector' in spec:
        headless_driver.find_element(By.CSS_SELECTOR, spec['setup_selector']).click()
        time.sleep(0.05)  # Brief pause for animation to start

    element = headless_driver.find_element(By.CSS_SELECTOR, spec['selector'])
    result = headless_driver.execute_script(HOVER_STABILITY_SCRIPT, element)

    # Check stability signals if specified
    signals = spec['expect'].get('stability_signals', {})

    if 'position_stable' in signals:
        assert result['stable'] == signals['position_stable'], (
            f'Expected stable={signals["position_stable"]}, got {result["stable"]}\nResult: {result}'
        )

    if 'animations_running' in signals:
        assert result['runningAnimations'] == signals['animations_running'], (
            f'Expected {signals["animations_running"]} running animations, '
            f'got {result["runningAnimations"]}\nResult: {result}'
        )

    if 'has_infinite_animation' in signals:
        assert result['hasInfiniteAnimation'] == signals['has_infinite_animation'], (
            f'Expected hasInfiniteAnimation={signals["has_infinite_animation"]}, '
            f'got {result["hasInfiniteAnimation"]}\nResult: {result}'
        )

    if 'final_distance_px' in signals:
        constraint = signals['final_distance_px']
        if isinstance(constraint, str) and constraint.startswith('<'):
            threshold = float(constraint[1:])
            assert result['finalDistance'] < threshold, (
                f'Expected finalDistance < {threshold}, got {result["finalDistance"]}\nResult: {result}'
            )


# ============================================================================
# VISIBILITY TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _visibility_specs, ids=lambda tc: tc['id'])
def test_hover_visibility(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test visibility detection for hidden/visible elements.

    Runs the full pre-hover pipeline (visibility + occlusion) since some
    elements pass visibility but fail occlusion (e.g., pointer-events: none).
    """
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')

    result = check_hover_actionability(headless_driver, spec['selector'])
    _assert_hover_result(result, spec['expect'])


# ============================================================================
# OCCLUSION TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _occlusion_specs, ids=lambda tc: tc['id'])
def test_hover_occlusion(
    headless_driver: webdriver.Chrome,
    examples_server: str,
    test_case: dict[str, Any],
) -> None:
    """Test occlusion detection (elementFromPoint check)."""
    spec = test_case['spec']
    headless_driver.get(f'{examples_server}/{spec["fixture"]}')

    result = check_hover_actionability(
        headless_driver,
        spec['selector'],
        setup_selector=spec.get('setup_selector'),
    )
    _assert_hover_result(result, spec['expect'])


# ============================================================================
# DURATION VALIDATION TESTS
# ============================================================================


@pytest.mark.parametrize('test_case', _duration_specs, ids=lambda tc: tc['id'])
def test_hover_duration_validation(test_case: dict[str, Any]) -> None:
    """Test duration_ms parameter validation via production code (no browser needed)."""
    spec = test_case['spec']
    duration_ms = spec.get('duration_ms', 0)
    expect = spec['expect']

    if expect['success']:
        validate_hover_duration(duration_ms)  # should not raise
    else:
        with pytest.raises(ValueError) as exc_info:
            validate_hover_duration(duration_ms)
        expected_error = expect.get('error_contains', '')
        assert expected_error in str(exc_info.value), (
            f'Expected error containing {expected_error!r}, got {str(exc_info.value)!r}'
        )


# -- Private Helpers ----------------------------------------------------------


def _assert_hover_result(result: Mapping[str, Any], expect: Mapping[str, Any]) -> None:
    """Assert hover check result matches YAML expectation."""
    if expect['success']:
        assert result['success'], f'Expected hover to succeed but got error: {result.get("error")}'
    else:
        assert not result['success'], 'Expected hover to fail but it succeeded'
        if 'error_contains' in expect:
            error = result.get('error', '')
            assert expect['error_contains'] in error, (
                f'Expected error containing {expect["error_contains"]!r}, got {error!r}'
            )
