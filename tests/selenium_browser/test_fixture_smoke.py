"""YAML-driven smoke tests for HTML fixtures with zero prior test coverage.

Validates ARIA tree snapshots for shadow DOM, role mapping, context-dependent
roles, nameless link URL fallback, forms, tables, iframes, hidden elements,
interactive elements, and error handling fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.selenium_browser.helpers import (
    TreeTestRunner,
    extract_assertions,
    load_yaml_test_specs,
)

_YAML_PATH = Path(__file__).parent / 'fixture_smoke_tests.yaml'
_all_specs = load_yaml_test_specs(_YAML_PATH)


@pytest.mark.parametrize('test_case', _all_specs, ids=lambda tc: tc['id'])
def test_fixture_smoke(
    tree_runner: TreeTestRunner,
    test_case: dict[str, Any],
) -> None:
    """Run ARIA tree snapshot against fixture and assert includes/excludes."""
    spec = test_case['spec']
    includes, excludes = extract_assertions(spec)

    tree_runner.assert_tree(
        fixture_path=spec['fixture'],
        includes=includes,
        excludes=excludes,
        selector=spec.get('selector', 'body'),
        include_urls=spec.get('include_urls', False),
        include_hidden=spec.get('include_hidden', False),
        include_page_info=spec.get('include_page_info', False),
    )
