"""Integration tests for SVG href handling in the aria/visual tree scripts."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import selenium_browser
from selenium import webdriver

_SCRIPTS_DIR = Path(selenium_browser.__file__).parent / 'scripts'
_SCRIPTS = {
    'aria': (_SCRIPTS_DIR / 'aria_snapshot.js').read_text(),
    'visual': (_SCRIPTS_DIR / 'visual_tree.js').read_text(),
}


class TestTreeSvgHref:
    """SVG links must serialize node.url as the href string, not the raw SVGAnimatedString object."""

    FIXTURE = """<a href="https://example.com/html">html</a>
<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
  <a href="https://example.com/svg"><circle cx="25" cy="25" r="20" /></a>
</svg>"""

    @pytest.mark.parametrize('script_name', ['aria', 'visual'])
    def test_link_url_is_string(self, headless_driver: webdriver.Chrome, script_name: str) -> None:
        encoded = base64.b64encode(self.FIXTURE.encode()).decode()
        headless_driver.get(f'data:text/html;base64,{encoded}')

        result = headless_driver.execute_script(_SCRIPTS[script_name], 'body', True, False)
        urls = _link_urls(result['tree'])

        # Pre-fix the SVG link's url was an SVGAnimatedString object, not a string.
        assert urls, 'expected at least one link url'
        assert all(isinstance(u, str) for u in urls)
        assert 'https://example.com/svg' in urls


# -- helpers -------------------------------------------------------------------


def _link_urls(node: dict[str, Any]) -> list[Any]:
    """Collect every node.url present in the tree."""
    urls: list[Any] = []
    if node.get('url') is not None:
        urls.append(node['url'])
    for child in node.get('children', []):
        urls.extend(_link_urls(child))
    return urls
