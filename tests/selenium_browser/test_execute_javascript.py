"""YAML-driven integration tests for execute_javascript tool.

Executes JavaScript in headless Chrome via the same async script pipeline
the MCP server uses, then asserts against expected result fields defined
in execute_javascript_tests.yaml.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium_browser.scripts import build_execute_javascript_async_script

from tests.selenium_browser.helpers import (
    assert_structured_result,
    load_yaml_test_specs_nested,
)

_YAML_PATH = Path(__file__).parent / 'execute_javascript_tests.yaml'
_all_specs = load_yaml_test_specs_nested(_YAML_PATH)

# Partition specs: timeout tests need script timeout adjustment
_standard_specs = [s for s in _all_specs if s['section'] != 'timeout_tests']
_timeout_specs = [s for s in _all_specs if s['section'] == 'timeout_tests']


# -- STANDARD TESTS (primitives, objects, promises, special types, etc.) -------


@pytest.mark.parametrize(
    'test_case',
    _standard_specs,
    ids=lambda tc: tc['id'],
)
def test_execute_javascript(headless_driver: webdriver.Chrome, test_case: dict[str, Any]) -> None:
    """Run execute_javascript test cases from YAML spec."""
    spec = test_case['spec']
    _navigate_to_fixture(headless_driver, spec)

    result = _execute_js(headless_driver, spec['code'])
    assert_structured_result(result, spec['expect'])


# -- TIMEOUT TESTS (need reduced script timeout) -------------------------------


@pytest.mark.parametrize(
    'test_case',
    _timeout_specs,
    ids=lambda tc: tc['id'],
)
def test_execute_javascript_timeout(headless_driver: webdriver.Chrome, test_case: dict[str, Any]) -> None:
    """Run timeout test cases using Selenium's native script timeout.

    ChromeDriver enforces script timeouts at the protocol level. For async
    scripts whose callback never fires, it raises ScriptTimeoutException.
    For synchronous infinite loops, ChromeDriver sends Runtime.terminateExecution
    to V8, which catches it at loop back-edge stack guards.

    The driver remains usable after ScriptTimeoutException per the W3C spec.
    """
    spec = test_case['spec']
    _navigate_to_fixture(headless_driver, spec)

    timeout_ms = spec.get('timeout_ms', 1000)
    expect = spec['expect']

    original_timeout = headless_driver.timeouts.script
    headless_driver.set_script_timeout(timeout_ms / 1000)

    try:
        result = _execute_js(headless_driver, spec['code'])
        # JS-level wrapper may have caught the timeout
        assert_structured_result(result, expect)
    except TimeoutException:
        # ChromeDriver-level timeout — expected for these tests
        assert expect.get('success') is False, 'Expected success=false for timeout test'
        assert expect.get('error_type') == 'timeout', 'Expected error_type=timeout'
    finally:
        headless_driver.set_script_timeout(original_timeout if original_timeout else 30)


# -- PRIVATE HELPERS -----------------------------------------------------------


def _execute_js(driver: webdriver.Chrome, code: str) -> dict[str, Any]:
    """Execute JavaScript through the same pipeline as the MCP server.

    Returns the raw result dict (matches JavaScriptResult fields).
    """
    escaped_code = json.dumps(code)
    async_script = build_execute_javascript_async_script(escaped_code)
    return driver.execute_async_script(async_script)


def _navigate_to_fixture(driver: webdriver.Chrome, spec: dict[str, Any]) -> None:
    """Navigate to the appropriate page for a test spec."""
    page = spec.get('page')
    fixture = spec.get('fixture')

    if fixture:
        # Inline HTML fixture — encode as data URL
        # Use base64 to handle all special characters (#, etc.)
        encoded = base64.b64encode(fixture.encode()).decode()
        driver.get(f'data:text/html;base64,{encoded}')
    elif page:
        driver.get(page)
