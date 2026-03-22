"""Integration test harness for selenium-browser-automation MCP server.

Demonstrates:
1. Headless Chrome via Selenium in pytest
2. Serving HTML fixtures via a threaded HTTP server
3. Executing the aria_snapshot.js script directly against the driver
4. YAML-driven parametrized testing with includes/excludes assertions

Key architectural insight: The MCP tool functions are closures over a
BrowserService instance. For testing, we bypass the MCP layer entirely and
call the same JavaScript scripts + Python post-processing code that the
tool functions use internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from selenium import webdriver
from selenium_browser_automation.tree_utils import serialize_aria_snapshot

from tests.selenium_browser_automation.helpers import (
    ARIA_SNAPSHOT_SCRIPT,
    VISUAL_TREE_SCRIPT,
    TreeTestRunner,
    assert_includes_excludes,
    collect_roles,
    extract_assertions,
    load_yaml_test_specs,
)

# ============================================================================
# PROTOTYPE 1: Minimal working test - direct JS execution
# ============================================================================


class TestMinimalSeleniumIntegration:
    """Prove that headless Chrome + JS script execution works."""

    def test_driver_starts(self, headless_driver: webdriver.Chrome) -> None:
        """Verify headless Chrome is running."""
        assert headless_driver.title is not None

    def test_navigate_data_url(self, headless_driver: webdriver.Chrome) -> None:
        """Navigate to a data: URL (no server needed)."""
        html = '<html><body><button>Click Me</button><a href="#">Link</a></body></html>'
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')
        assert 'Click Me' in headless_driver.page_source

    def test_aria_snapshot_script_basic(self, headless_driver: webdriver.Chrome) -> None:
        """Execute aria_snapshot.js and verify output structure."""
        html = """
        <html><body>
            <nav aria-label="Main">
                <a href="/">Home</a>
                <a href="/about">About</a>
            </nav>
            <main>
                <h1>Welcome</h1>
                <button>Submit</button>
            </main>
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)

        assert result is not None
        assert 'tree' in result
        assert 'stats' in result

        tree = result['tree']
        assert tree is not None
        assert tree.get('role') is not None

        stats = result['stats']
        assert 'links' in stats

    def test_aria_snapshot_includes_expected_roles(self, headless_driver: webdriver.Chrome) -> None:
        """Verify the ARIA tree contains expected semantic roles."""
        html = """
        <html><body>
            <h1>Title</h1>
            <button>Click</button>
            <input type="checkbox" aria-label="Agree">
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)
        tree = result['tree']

        roles = collect_roles(tree)
        assert 'heading' in roles
        assert 'button' in roles
        assert 'checkbox' in roles

    def test_visual_tree_script_basic(self, headless_driver: webdriver.Chrome) -> None:
        """Execute visual_tree.js and verify output structure."""
        html = """
        <html><body>
            <button>Visible</button>
            <div style="display:none">Hidden</div>
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(VISUAL_TREE_SCRIPT, 'body', False, False)

        assert result is not None
        assert 'tree' in result
        assert 'stats' in result


class TestFixtureServer:
    """Verify the HTTP fixture server works."""

    def test_server_serves_files(self, headless_driver: webdriver.Chrome, examples_server: str) -> None:
        """Navigate to an HTML fixture served via HTTP."""
        headless_driver.get(f'{examples_server}/examples/test_error_handling.html')
        assert headless_driver.title


# ============================================================================
# PROTOTYPE 2: YAML-driven test spec runner
# ============================================================================


# Load the include_hidden test specs for parametrization
_INCLUDE_HIDDEN_YAML = Path(__file__).parent / 'include_hidden_tests.yaml'
_include_hidden_specs: list[dict[str, Any]] = []
if _INCLUDE_HIDDEN_YAML.exists():
    _include_hidden_specs = load_yaml_test_specs(_INCLUDE_HIDDEN_YAML)


class TestYamlSpecParsing:
    """Verify the YAML spec loading and assertion framework works."""

    def test_load_include_hidden_specs(self) -> None:
        """Verify we can parse the include_hidden_tests.yaml file."""
        assert len(_include_hidden_specs) > 0, 'No test specs found'

        for case in _include_hidden_specs:
            assert 'id' in case
            assert 'name' in case
            assert 'spec' in case

    def test_load_compact_tree_specs(self) -> None:
        """Verify we can parse the compact_tree_tests.yaml file."""
        yaml_path = Path(__file__).parent / 'compact_tree_tests.yaml'
        specs = load_yaml_test_specs(yaml_path)
        assert len(specs) > 0

        test_1_1 = next((s for s in specs if s['id'] == '1.1'), None)
        assert test_1_1 is not None
        assert test_1_1['name'] == 'simple_empty_div'

    def test_assert_includes_excludes_passing(self) -> None:
        """Test the assertion helper with passing conditions."""
        output = '- button "Submit"\n- heading "Title" [level=1]'
        assert_includes_excludes(
            output,
            includes=['button "Submit"', 'heading "Title"'],
            excludes=['generic', 'text: Submit'],
        )

    def test_assert_includes_excludes_missing(self) -> None:
        """Test the assertion helper detects missing includes."""
        output = '- button "Submit"'
        with pytest.raises(AssertionError, match='MISSING'):
            assert_includes_excludes(output, includes=['NOT HERE'], excludes=[])

    def test_assert_includes_excludes_forbidden(self) -> None:
        """Test the assertion helper detects forbidden strings."""
        output = '- generic\n- button "Submit"'
        with pytest.raises(AssertionError, match='FOUND'):
            assert_includes_excludes(output, includes=[], excludes=['generic'])


# ============================================================================
# PROTOTYPE 3: End-to-end ARIA snapshot with data: URL
# ============================================================================


class TestAriaSnapshotEndToEnd:
    """Full pipeline: HTML -> headless Chrome -> JS script -> serialize -> assert."""

    def test_simple_form(self, headless_driver: webdriver.Chrome) -> None:
        """Test ARIA snapshot of a simple form."""
        html = """
        <html><body>
            <form aria-label="Login">
                <label for="user">Username</label>
                <input id="user" type="text">
                <label for="pass">Password</label>
                <input id="pass" type="password">
                <button type="submit">Sign In</button>
            </form>
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'form', False, False)
        assert result is not None

        tree = result['tree']
        output = serialize_aria_snapshot(tree)

        assert_includes_excludes(
            output,
            includes=[
                'form "Login"',
                'textbox "Username"',
                'button "Sign In"',
            ],
            excludes=[],
        )

    def test_checkbox_states(self, headless_driver: webdriver.Chrome) -> None:
        """Test ARIA snapshot captures checkbox checked state."""
        html = """
        <html><body>
            <input type="checkbox" aria-label="Checked" checked>
            <input type="checkbox" aria-label="Unchecked">
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)
        output = serialize_aria_snapshot(result['tree'])

        assert '[checked]' in output
        assert '[unchecked]' in output

    def test_hidden_elements_excluded(self, headless_driver: webdriver.Chrome) -> None:
        """Test that hidden elements are excluded from ARIA tree."""
        html = """
        <html><body>
            <button>Visible</button>
            <button aria-hidden="true">Hidden</button>
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)
        output = serialize_aria_snapshot(result['tree'])

        assert 'button "Visible"' in output
        assert 'Hidden' not in output

    def test_hidden_elements_included(self, headless_driver: webdriver.Chrome) -> None:
        """Test that include_hidden=True shows hidden elements with markers."""
        html = """
        <html><body>
            <button>Visible</button>
            <button aria-hidden="true">Hidden</button>
        </body></html>
        """
        headless_driver.get(f'data:text/html;charset=utf-8,{html}')

        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, True)
        output = serialize_aria_snapshot(result['tree'])

        assert 'button "Visible"' in output
        assert 'Hidden' in output
        assert '[hidden:aria-hidden]' in output


# ============================================================================
# YAML-DRIVEN PARAMETRIZED TESTS
# ============================================================================


@pytest.mark.parametrize(
    'test_case',
    load_yaml_test_specs(Path(__file__).parent / 'compact_tree_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_compact_tree_spec(tree_runner: TreeTestRunner, test_case: dict[str, Any]) -> None:
    """Run all compact_tree_tests.yaml test cases."""
    spec = test_case['spec']

    selector = spec.get('selector', 'body')
    fixture_path = spec.get('fixture', 'examples/compact-tree.html')

    if 'compact' in spec:
        includes, excludes = extract_assertions(
            spec['compact'] if isinstance(spec.get('compact'), dict) else spec,
            'tree',
        )
        tree_runner.assert_tree(
            fixture_path=fixture_path,
            selector=selector,
            tool_name='aria',
            compact_tree=True,
            includes=includes,
            excludes=excludes,
        )

    if 'full' in spec:
        includes, excludes = extract_assertions(spec['full'] if isinstance(spec.get('full'), dict) else spec, 'tree')
        tree_runner.assert_tree(
            fixture_path=fixture_path,
            selector=selector,
            tool_name='aria',
            compact_tree=False,
            includes=includes,
            excludes=excludes,
        )


@pytest.mark.parametrize(
    'test_case',
    load_yaml_test_specs(Path(__file__).parent / 'include_hidden_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_include_hidden_spec(tree_runner: TreeTestRunner, test_case: dict[str, Any]) -> None:
    """Run all include_hidden_tests.yaml test cases."""
    spec = test_case['spec']
    includes, excludes = extract_assertions(spec, 'tree')

    tree_runner.assert_tree(
        fixture_path=spec.get('fixture', 'examples/include-hidden-test.html'),
        selector=spec.get('selector', 'body'),
        tool_name='aria',
        include_hidden=spec.get('include_hidden', False),
        compact_tree=False,
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    load_yaml_test_specs(Path(__file__).parent / 'visual_tree_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_visual_tree_spec(tree_runner: TreeTestRunner, test_case: dict[str, Any]) -> None:
    """Run all visual_tree_tests.yaml test cases."""
    spec = test_case['spec']
    includes, excludes = extract_assertions(spec, 'tree')

    tree_runner.assert_tree(
        fixture_path=spec.get('fixture', 'examples/visual-tree-test.html'),
        selector=spec.get('selector', 'body'),
        tool_name='visual',
        include_hidden=spec.get('include_hidden', False),
        compact_tree=spec.get('compact_tree', False),
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    load_yaml_test_specs(Path(__file__).parent / 'state_detection_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_state_detection_spec(tree_runner: TreeTestRunner, test_case: dict[str, Any]) -> None:
    """Run all state_detection_tests.yaml test cases."""
    spec = test_case['spec']
    includes, excludes = extract_assertions(spec, 'tree')

    tree_runner.assert_tree(
        fixture_path=spec.get('fixture', 'examples/state-detection.html'),
        selector=spec.get('selector', 'body'),
        tool_name='aria',
        include_hidden=spec.get('include_hidden', False),
        compact_tree=spec.get('compact_tree', False),
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    load_yaml_test_specs(Path(__file__).parent / 'page_metadata_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_page_metadata_spec(tree_runner: TreeTestRunner, test_case: dict[str, Any]) -> None:
    """Run all page_metadata_tests.yaml scenarios.

    These tests verify the two-tier metadata footer system by asserting
    tree body and metadata footer independently.
    """
    spec = test_case['spec']

    tool_field = spec.get('tool', 'get_aria_snapshot')
    tool_name = 'visual' if 'visual' in tool_field.lower() else 'aria'
    args = spec.get('args', {})

    tree_yaml, metadata_footer = tree_runner.snapshot(
        fixture_path=spec.get('fixture', 'examples/test_page_metadata_integration.html'),
        selector=args.get('selector', 'body'),
        tool_name=tool_name,
        include_urls=args.get('include_urls', False),
        include_hidden=args.get('include_hidden', False),
        compact_tree=args.get('compact_tree', True),
        include_page_info=args.get('include_page_info', False),
    )

    meta_includes, meta_excludes = extract_assertions(spec, 'metadata')
    if meta_includes or meta_excludes:
        assert_includes_excludes(metadata_footer, meta_includes, meta_excludes)

    tree_includes, tree_excludes = extract_assertions(spec, 'tree')
    if tree_includes or tree_excludes:
        assert_includes_excludes(tree_yaml, tree_includes, tree_excludes)
