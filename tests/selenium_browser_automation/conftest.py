"""Shared pytest fixtures for selenium-browser-automation tests."""

from __future__ import annotations

import http.server
import threading
from pathlib import Path
from typing import Any

import pytest
import selenium_browser_automation
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from tests.selenium_browser_automation.helpers import TreeTestRunner

# Resolve paths for serving fixtures via HTTP
_SELENIUM_PROJECT_ROOT = Path(selenium_browser_automation.__file__).parent.parent
_WORKSPACE_ROOT = _SELENIUM_PROJECT_ROOT.parent.parent


@pytest.fixture(scope='session')
def headless_driver() -> Any:
    """Session-scoped headless Chrome/Chromium driver."""
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1920,1080')
    opts.add_argument('--disable-gpu')

    chromium_path = '/Applications/Chromium.app/Contents/MacOS/Chromium'
    if Path(chromium_path).exists():
        opts.binary_location = chromium_path

    driver = webdriver.Chrome(options=opts)
    yield driver
    driver.quit()


@pytest.fixture(scope='session')
def examples_server() -> Any:
    """Session-scoped HTTP server serving from the selenium project root.

    Serves files under mcp/selenium-browser-automation/ (e.g. 'examples/compact-tree.html').
    """

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(_SELENIUM_PROJECT_ROOT), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f'http://127.0.0.1:{port}'

    server.shutdown()


@pytest.fixture(scope='session')
def tree_runner(headless_driver: webdriver.Chrome, examples_server: str) -> TreeTestRunner:
    """Session-scoped tree test runner encapsulating driver + server."""
    return TreeTestRunner(headless_driver, examples_server)
