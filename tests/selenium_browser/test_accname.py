"""YAML-driven tests for accessible name computation (AccName 1.2 spec compliance).

Red-green TDD: tests written BEFORE the fix to expose bugs, then verified
after the fix to confirm correctness.

Covers: priority order (2B/2D), host-language labels (2E: label, legend,
figcaption, caption), and title as last resort (2I).
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

_YAML_PATH = Path(__file__).parent / 'accname_tests.yaml'
_all_specs = load_yaml_test_specs(_YAML_PATH)


@pytest.mark.parametrize('test_case', _all_specs, ids=lambda tc: tc['id'])
def test_accname(
    tree_runner: TreeTestRunner,
    test_case: dict[str, Any],
) -> None:
    """Run accessible name computation test against fixture."""
    spec = test_case['spec']
    includes, excludes = extract_assertions(spec)

    tree_runner.assert_tree(
        fixture_path=spec['fixture'],
        includes=includes,
        excludes=excludes,
        selector=spec.get('selector', 'body'),
    )
