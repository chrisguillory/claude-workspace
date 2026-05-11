"""Shared input validation for Selenium Browser Automation.

Framework-free validators that raise ValueError. MCP tool wrappers
convert to ToolError at the boundary.
"""

from __future__ import annotations

import re
from typing import NewType

__all__ = [
    'NON_CSS_SELECTOR_RE',
    'CssSelector',
    'validate_css_selector',
]


# Brand returned by ``validate_css_selector``. Functions that ultimately call
# Selenium's ``find_element*`` accept this type so the type checker rejects
# raw user input that hasn't been screened for Playwright-style selectors.
CssSelector = NewType('CssSelector', str)


NON_CSS_SELECTOR_RE = re.compile(
    r':(?:has-text|text|nth-match)\s*\('
    r'|:(?:visible|hidden)\b'
    r'|\b(?:text|role|data-testid)\s*='
    r'|>>',
)


def validate_css_selector(selector: str) -> CssSelector:
    """Validate CSS selector syntax and return the branded value."""
    match = NON_CSS_SELECTOR_RE.search(selector)
    if match:
        raise ValueError(
            f"Selector '{selector}' contains non-CSS syntax ('{match.group()}'). "
            f'Only standard CSS selectors are supported.\n'
            f"To find elements by text: get_interactive_elements(text_contains='...')\n"
            f"To discover selectors: get_aria_snapshot('body', include_urls=True)",
        )
    return CssSelector(selector)
