"""Synchronous hover actionability checks for Selenium WebDriver.

Extracted from the MCP tool for direct use in tests and the async wrapper.
Performs visibility and occlusion checks before hover actions.
"""

from __future__ import annotations

import time

from cc_lib.types import JsonObject
from selenium import webdriver
from selenium.webdriver.common.by import By

from .scripts import HOVER_OCCLUSION_SCRIPT

__all__ = ['check_hover_actionability', 'validate_hover_duration']


def check_hover_actionability(
    driver: webdriver.Chrome,
    css_selector: str,
    *,
    setup_selector: str | None = None,
) -> JsonObject:
    """Check whether an element is actionable for hover.

    Performs: optional setup click, find element, scroll into view,
    is_displayed check, and occlusion check via elementFromPoint.

    Returns dict with 'success' bool and optional 'error' string.
    """
    if setup_selector:
        driver.find_element(By.CSS_SELECTOR, setup_selector).click()
        time.sleep(0.2)  # Let UI state settle

    element = driver.find_element(By.CSS_SELECTOR, css_selector)

    # Scroll into view (matches server behavior)
    driver.execute_script(
        "arguments[0].scrollIntoView({behavior: 'instant', block: 'nearest'});",
        element,
    )

    # Visibility check (matches server: element.is_displayed())
    if not element.is_displayed():
        return {'success': False, 'error': 'not visible'}

    # Occlusion check via elementFromPoint at center.
    # Script computes its own center via getBoundingClientRect()
    # (viewport-relative, which is what elementFromPoint requires).
    result = driver.execute_script(HOVER_OCCLUSION_SCRIPT, element)
    if result == 'obscured':
        return {'success': False, 'error': 'obscured'}

    return {'success': True}


def validate_hover_duration(duration_ms: int) -> None:
    """Validate hover duration_ms parameter. Raises ValueError if out of range."""
    if duration_ms < 0:
        raise ValueError('duration_ms cannot be negative')
    if duration_ms > 30000:
        raise ValueError('duration_ms exceeds maximum of 30000ms (30 seconds)')
