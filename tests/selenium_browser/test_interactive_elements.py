"""Integration tests for get_interactive_elements (headless browser)."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence

from selenium import webdriver
from selenium_browser.models import InteractiveElement
from selenium_browser.service import BrowserService
from selenium_browser.state import BrowserState


class TestGetInteractiveElements:
    """Drives the real BrowserService method end-to-end against a headless browser."""

    # An SVG <a> exposes `href` as an SVGAnimatedString (an object), not a string.
    # Emitting that straight into InteractiveElement.href (str | None) used to crash
    # the tool on any page with an interactive SVG (e.g. a clickable logo).
    SVG_HREF_FIXTURE = """<!doctype html>
<html><body>
  <a href="https://example.com/html-link" style="cursor:pointer">HTML link</a>
  <svg width="120" height="120" xmlns="http://www.w3.org/2000/svg">
    <a href="https://example.com/svg-link" style="cursor:pointer">
      <circle cx="60" cy="60" r="50" fill="#1f6feb" />
    </a>
  </svg>
</body></html>"""

    def test_handles_svg_href(self, headless_driver: webdriver.Chrome) -> None:
        encoded = base64.b64encode(self.SVG_HREF_FIXTURE.encode()).decode()
        headless_driver.get(f'data:text/html;base64,{encoded}')

        # Pre-fix, this call raised pydantic.ValidationError (SVG href is an object).
        elements = asyncio.run(_collect(headless_driver))

        assert all(e.href is None or isinstance(e.href, str) for e in elements)
        hrefs = {e.href for e in elements}
        assert 'https://example.com/html-link' in hrefs
        assert 'https://example.com/svg-link' in hrefs


# -- helpers -------------------------------------------------------------------


async def _collect(driver: webdriver.Chrome) -> Sequence[InteractiveElement]:
    """Run the real get_interactive_elements on the driver's current page."""
    state = await BrowserState.create()
    state.driver = driver  # get_browser() returns an already-set driver as-is
    try:
        return await BrowserService(state).get_interactive_elements('body')
    finally:
        # Release the factory's temp dirs without quitting the shared fixture driver.
        state.temp_dir.cleanup()
        state.capture_temp_dir.cleanup()
