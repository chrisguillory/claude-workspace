"""YAML-driven integration tests for hover tool actionability checks.

Tests the hover tool's pre-hover pipeline: duration validation, visibility
detection, stability analysis, and occlusion detection. Each test category
exercises a different stage of the pipeline using the same JavaScript checks
the MCP server executes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By

from tests.selenium_browser_automation.helpers import load_yaml_test_specs

_YAML_PATH = Path(__file__).parent / 'hover_tests.yaml'
_all_specs = [s for s in load_yaml_test_specs(_YAML_PATH) if s['spec'].get('status') != 'planned']

# Partition by section
_stability_specs = [s for s in _all_specs if s['section'] == 'stability']
_visibility_specs = [s for s in _all_specs if s['section'] == 'visibility']
_occlusion_specs = [s for s in _all_specs if s['section'] == 'occlusion']
_duration_specs = [s for s in _all_specs if s['section'] == 'duration_validation']

# Occlusion check via elementFromPoint — mirrors server.py hover() occlusion logic.
# Uses getBoundingClientRect() for viewport-relative coordinates (elementFromPoint
# requires viewport coords). The server computes center via element.rect in Python,
# but both approaches produce identical results after scrollIntoView.
_OCCLUSION_CHECK_JS = """
const target = arguments[0];
const rect = target.getBoundingClientRect();
const x = rect.left + rect.width / 2;
const y = rect.top + rect.height / 2;
const atPoint = document.elementFromPoint(x, y);
if (!atPoint) return 'no_element';
if (atPoint === target) return 'ok';
if (target.contains(atPoint)) return 'ok';
return 'obscured';
"""

# Multi-signal stability check — mirrors server.py hover() stability logic.
# Uses requestAnimationFrame for position polling + Web Animations API for
# CSS animation detection. Returns a Promise that ChromeDriver auto-resolves.
# Keep in sync with hover() in server.py if the stability algorithm changes.
_STABILITY_CHECK_JS = """
const el = arguments[0];
const DISTANCE_THRESHOLD = 5;
const MAX_CHECKS = 10;

return new Promise((resolve) => {
    let animations = [];
    let hasInfiniteAnimation = false;
    try {
        animations = el.getAnimations();
        hasInfiniteAnimation = animations.some(a => {
            const effect = a.effect;
            if (effect && effect.getTiming) {
                const timing = effect.getTiming();
                return timing.iterations === Infinity;
            }
            return false;
        });
    } catch (e) {}

    const runningAnimations = animations.filter(a => a.playState === 'running');
    let prevRect = el.getBoundingClientRect();
    let checkCount = 0;
    let consecutiveStable = 0;

    function checkStability() {
        checkCount++;
        const currRect = el.getBoundingClientRect();
        const dx = currRect.x - prevRect.x;
        const dy = currRect.y - prevRect.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const sizeStable = (
            Math.abs(currRect.width - prevRect.width) < 1 &&
            Math.abs(currRect.height - prevRect.height) < 1
        );
        const isStable = distance < DISTANCE_THRESHOLD && sizeStable;

        if (isStable) {
            consecutiveStable++;
            if (consecutiveStable >= 2) {
                resolve({
                    stable: true,
                    framesChecked: checkCount,
                    runningAnimations: runningAnimations.length,
                    hasInfiniteAnimation: hasInfiniteAnimation,
                    finalDistance: distance
                });
                return;
            }
        } else {
            consecutiveStable = 0;
        }

        prevRect = currRect;

        if (checkCount >= MAX_CHECKS) {
            resolve({
                stable: false,
                framesChecked: checkCount,
                runningAnimations: runningAnimations.length,
                hasInfiniteAnimation: hasInfiniteAnimation,
                finalDistance: distance,
                reason: 'timeout'
            });
            return;
        }

        requestAnimationFrame(checkStability);
    }

    requestAnimationFrame(checkStability);
});
"""


# ============================================================================
# HOVER ACTIONABILITY CHECKS
# ============================================================================


def _check_hover_actionability(
    driver: webdriver.Chrome,
    selector: str,
    setup_selector: str | None = None,
) -> dict[str, Any]:
    """Replicate the hover tool's pre-hover checks (visibility + occlusion).

    Mirrors server.py hover() steps 3-6: scroll into view, check is_displayed,
    then check elementFromPoint for occlusion.

    Returns dict with 'success' bool and optional 'error' string.
    """
    if setup_selector:
        driver.find_element(By.CSS_SELECTOR, setup_selector).click()
        time.sleep(0.2)  # Let UI state settle

    element = driver.find_element(By.CSS_SELECTOR, selector)

    # Scroll into view (matches server behavior)
    driver.execute_script(
        "arguments[0].scrollIntoView({behavior: 'instant', block: 'nearest'});",
        element,
    )

    # Visibility check (matches server: element.is_displayed())
    if not element.is_displayed():
        return {'success': False, 'error': 'not visible'}

    # Occlusion check (matches server: elementFromPoint at center)
    result = driver.execute_script(_OCCLUSION_CHECK_JS, element)
    if result == 'obscured':
        return {'success': False, 'error': 'obscured'}

    return {'success': True}


def _assert_hover_result(result: dict[str, Any], expect: dict[str, Any]) -> None:
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
    result = headless_driver.execute_script(_STABILITY_CHECK_JS, element)

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

    result = _check_hover_actionability(headless_driver, spec['selector'])
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

    result = _check_hover_actionability(
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
    """Test duration_ms parameter validation (no browser needed).

    Replicates the server's duration validation logic:
    - duration_ms < 0: rejected ("cannot be negative")
    - duration_ms > 30000: rejected ("exceeds maximum")
    - 0 <= duration_ms <= 30000: valid
    """
    spec = test_case['spec']
    duration_ms = spec.get('duration_ms', 0)
    expect = spec['expect']

    # Replicates hover() duration validation
    error_msg = None
    if duration_ms < 0:
        error_msg = 'cannot be negative'
    elif duration_ms > 30000:
        error_msg = 'exceeds maximum'

    if expect['success']:
        assert error_msg is None, f'Expected duration_ms={duration_ms} to be valid, but got error: {error_msg}'
    else:
        assert error_msg is not None, f'Expected duration_ms={duration_ms} to be invalid'
        expected_error = expect.get('error_contains', '')
        assert expected_error in error_msg, f'Expected error containing {expected_error!r}, got {error_msg!r}'
