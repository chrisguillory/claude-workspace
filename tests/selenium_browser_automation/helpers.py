"""Shared test utilities: YAML loaders, assertion helpers, and tree test runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import selenium_browser_automation
import yaml
from selenium import webdriver
from selenium_browser_automation.helpers import build_page_metadata, count_tree_nodes
from selenium_browser_automation.server import (
    _ARIA_HIDDEN_REASON_KEYS,
    _VISUAL_HIDDEN_REASON_KEYS,
)
from selenium_browser_automation.tree_utils import (
    compact_aria_tree,
    compact_visual_tree,
    serialize_aria_snapshot,
    serialize_visual_tree,
)

# Load the JavaScript scripts at module level (same as the MCP server does)
_SCRIPTS_DIR = Path(selenium_browser_automation.__file__).parent / 'scripts'
ARIA_SNAPSHOT_SCRIPT = (_SCRIPTS_DIR / 'aria_snapshot.js').read_text()
VISUAL_TREE_SCRIPT = (_SCRIPTS_DIR / 'visual_tree.js').read_text()


# -- TREE TEST RUNNER ----------------------------------------------------------


class TreeTestRunner:
    """Encapsulates headless browser + HTTP server for tree snapshot testing.

    Owns the session-scoped driver and base URL, providing snapshot execution
    and assertion methods that eliminate per-test boilerplate.
    """

    def __init__(self, driver: webdriver.Chrome, base_url: str) -> None:
        self.driver = driver
        self.base_url = base_url

    def fixture_url(self, path: str) -> str:
        return f'{self.base_url}/{path}'

    def snapshot(
        self,
        fixture_path: str,
        selector: str = 'body',
        tool_name: str = 'aria',
        include_urls: bool = False,
        include_hidden: bool = False,
        compact_tree: bool = True,
        include_page_info: bool = False,
    ) -> tuple[str, str]:
        """Execute the full tree snapshot pipeline.

        Returns (tree_yaml, metadata_footer).
        """
        self.driver.get(self.fixture_url(fixture_path))

        is_aria = tool_name == 'aria'
        script = ARIA_SNAPSHOT_SCRIPT if is_aria else VISUAL_TREE_SCRIPT
        reason_keys = _ARIA_HIDDEN_REASON_KEYS if is_aria else _VISUAL_HIDDEN_REASON_KEYS

        result = self.driver.execute_script(script, selector, include_urls, include_hidden)

        if result is None:
            tree_data = None
            stats: dict[str, Any] = {}
        else:
            tree_data = result.get('tree')
            stats = result.get('stats', {})

        raw_count = count_tree_nodes(tree_data)

        if compact_tree and tree_data:
            tree_data = compact_aria_tree(tree_data) if is_aria else compact_visual_tree(tree_data)

        tree_yaml = serialize_aria_snapshot(tree_data) if is_aria else serialize_visual_tree(tree_data)

        compacted_count = count_tree_nodes(tree_data)
        metadata_footer = build_page_metadata(
            page_stats=stats,
            include_page_info=include_page_info,
            include_urls=include_urls,
            compact_tree=compact_tree,
            raw_node_count=raw_count,
            compacted_node_count=compacted_count,
            hidden_reason_keys=reason_keys,
        )

        return tree_yaml, metadata_footer or ''

    def assert_tree(
        self,
        fixture_path: str,
        includes: Sequence[str],
        excludes: Sequence[str],
        selector: str = 'body',
        tool_name: str = 'aria',
        include_urls: bool = False,
        include_hidden: bool = False,
        compact_tree: bool = True,
        include_page_info: bool = False,
    ) -> None:
        """Snapshot + assert includes/excludes on combined output."""
        tree_yaml, metadata_footer = self.snapshot(
            fixture_path=fixture_path,
            selector=selector,
            tool_name=tool_name,
            include_urls=include_urls,
            include_hidden=include_hidden,
            compact_tree=compact_tree,
            include_page_info=include_page_info,
        )
        combined = tree_yaml + metadata_footer
        assert_includes_excludes(combined, includes, excludes)


# -- YAML LOADERS --------------------------------------------------------------


def load_yaml_test_specs(yaml_path: Path) -> list[dict[str, Any]]:
    """Load test specs from a YAML file and flatten into parametrizable cases.

    Handles section > tests two-level nesting. Returns a list of dicts
    with: id, name, section, spec.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    metadata = data.get('metadata', {})
    global_defaults = {}
    if 'fixture' in metadata:
        global_defaults['fixture'] = metadata['fixture']

    cases: list[dict[str, Any]] = []

    for section_key, section in data.items():
        if section_key == 'metadata':
            continue
        if not isinstance(section, dict):
            continue

        tests = section.get('tests', [])
        if not tests and ('selector' in section or 'tool' in section):
            cases.append(
                {
                    'id': section_key,
                    'name': section_key,
                    'section': section_key,
                    'spec': {**global_defaults, **section},
                },
            )
        else:
            section_defaults = {k: v for k, v in section.items() if k not in ['tests', 'description']}
            cases.extend(
                {
                    'id': test.get('id', test.get('name', 'unknown')),
                    'name': test.get('name', 'unnamed'),
                    'section': section_key,
                    'spec': {**global_defaults, **section_defaults, **test},
                }
                for test in tests
            )

    return cases


def load_yaml_test_specs_nested(yaml_path: Path) -> list[dict[str, Any]]:
    """Load test specs from YAML with section > categories > tests 3-level nesting.

    Used for execute_javascript_tests.yaml which groups tests by category
    within each execution context (about_blank, dom_basic, etc.).

    Returns a list of dicts with: id, name, section, category, spec.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases: list[dict[str, Any]] = []

    for section_key, section in data.items():
        if section_key == 'metadata' or not isinstance(section, dict):
            continue

        # Section-level defaults (page, fixture, etc.)
        section_defaults = {k: v for k, v in section.items() if k not in ['categories', 'description', 'note']}

        categories = section.get('categories', {})
        if not isinstance(categories, dict):
            continue

        for cat_key, category in categories.items():
            if not isinstance(category, dict):
                continue

            cases.extend(
                {
                    'id': f'{section_key}.{cat_key}.{test["name"]}',
                    'name': test['name'],
                    'section': section_key,
                    'category': cat_key,
                    'spec': {**section_defaults, **test},
                }
                for test in category.get('tests', [])
            )

    return cases


# -- ASSERTION HELPERS ---------------------------------------------------------


def assert_includes_excludes(output: str, includes: Sequence[str], excludes: Sequence[str]) -> None:
    """Assert that output contains all includes and none of excludes."""
    for expected in includes:
        assert expected in output, f'MISSING expected string: {expected!r}\n--- Output ---\n{output[:2000]}'
    for forbidden in excludes:
        assert forbidden not in output, f'FOUND forbidden string: {forbidden!r}\n--- Output ---\n{output[:2000]}'


def extract_assertions(spec: dict[str, Any], mode: str = 'tree') -> tuple[list[str], list[str]]:
    """Extract includes/excludes from various YAML assertion formats.

    Args:
        spec: Test specification dict
        mode: 'tree' for tree assertions, 'metadata' for metadata assertions
    """
    for key in ['expect', 'compact', 'full', 'metadata', 'tree']:
        if key in spec:
            if (
                (mode == 'metadata' and key == 'metadata')
                or (mode == 'tree' and key == 'tree')
                or (mode == 'tree' and key in ['expect', 'compact', 'full'])
            ):
                return (
                    spec[key].get('includes', []),
                    spec[key].get('excludes', []),
                )
    return ([], [])


def collect_roles(node: dict[str, Any]) -> set[str]:
    """Recursively collect all roles from an ARIA tree."""
    roles: set[str] = set()
    if node is None:
        return roles
    if 'role' in node:
        roles.add(node['role'])
    for child in node.get('children', []):
        roles.update(collect_roles(child))
    return roles


def assert_structured_result(result: dict[str, Any], expect: dict[str, Any]) -> None:
    """Assert structured fields of a JavaScriptResult-like dict against expected values.

    Supports:
      - success: exact bool match
      - result: exact value match (when present)
      - result_type: exact string match
      - error_type: exact string match
      - note_contains: substring match on note field
    """
    if 'success' in expect:
        assert result['success'] == expect['success'], (
            f'Expected success={expect["success"]}, got {result["success"]}\nResult: {result}'
        )

    if 'result' in expect:
        assert result.get('result') == expect['result'], (
            f'Expected result={expect["result"]!r}, got {result.get("result")!r}\nResult: {result}'
        )

    if 'result_type' in expect:
        assert result.get('result_type') == expect['result_type'], (
            f'Expected result_type={expect["result_type"]!r}, got {result.get("result_type")!r}\nResult: {result}'
        )

    if 'error_type' in expect:
        assert result.get('error_type') == expect['error_type'], (
            f'Expected error_type={expect["error_type"]!r}, got {result.get("error_type")!r}\nResult: {result}'
        )

    if 'note_contains' in expect:
        note = result.get('note') or ''
        assert expect['note_contains'] in note, (
            f'Expected note to contain {expect["note_contains"]!r}, got {note!r}\nResult: {result}'
        )
