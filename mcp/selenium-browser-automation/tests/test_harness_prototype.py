"""Prototype: Integration test harness for selenium-browser-automation MCP server.

This prototype demonstrates:
1. Headless Chrome via Selenium in pytest
2. Serving HTML fixtures via a threaded HTTP server
3. Executing the aria_snapshot.js script directly against the driver
4. YAML-driven parametrized testing with includes/excludes assertions

Key architectural insight: The MCP tool functions are closures over a
BrowserService instance. For testing, we bypass the MCP layer entirely and
call the same JavaScript scripts + Python post-processing code that the
tool functions use internally.

Two viable testing approaches:
A) "Script-level" - Execute JS scripts directly via driver.execute_script()
   and assert on the raw JSON output. Tests the JS logic.
B) "Tool-level" - Construct a BrowserService, inject a test driver, and call
   the tool implementation. Tests the full Python+JS pipeline.

This prototype implements approach A as proof-of-concept.
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Load the JavaScript scripts at module level (same as the MCP server does)
_SCRIPTS_DIR = Path(__file__).parent.parent / 'selenium_browser_automation' / 'scripts'
ARIA_SNAPSHOT_SCRIPT = (_SCRIPTS_DIR / 'aria_snapshot.js').read_text()
VISUAL_TREE_SCRIPT = (_SCRIPTS_DIR / 'visual_tree.js').read_text()


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope='session')
def headless_driver() -> Any:
    """Session-scoped headless Chrome/Chromium driver.

    Uses Chromium on macOS if available, falls back to Chrome.
    Session-scoped to avoid the ~2s startup cost per test.
    """
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1920,1080')
    opts.add_argument('--disable-gpu')

    # Use Chromium if available (matches production server behavior)
    chromium_path = '/Applications/Chromium.app/Contents/MacOS/Chromium'
    if Path(chromium_path).exists():
        opts.binary_location = chromium_path

    driver = webdriver.Chrome(options=opts)
    yield driver
    driver.quit()


@pytest.fixture(scope='session')
def fixture_server() -> Any:
    """Session-scoped HTTP server for HTML fixture files.

    Serves files from the tests/ directory on an ephemeral port.
    Using HTTP (not file://) ensures proper security context for JS execution.
    """
    tests_dir = Path(__file__).parent

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        """HTTP handler that suppresses access logs."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(tests_dir), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # Suppress request logging during tests

    server = http.server.HTTPServer(('127.0.0.1', 0), QuietHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f'http://127.0.0.1:{port}'

    server.shutdown()


# ============================================================================
# PROTOTYPE 1: Minimal working test - direct JS execution
# ============================================================================


class TestMinimalSeleniumIntegration:
    """Prove that headless Chrome + JS script execution works."""

    def test_driver_starts(self, headless_driver: webdriver.Chrome) -> None:
        """Verify headless Chrome is running."""
        assert headless_driver.title is not None  # Driver is alive

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

        # Execute the same script the MCP tool uses
        # Args: selector, includeUrls, includeHidden
        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)

        # Verify structure
        assert result is not None
        assert 'tree' in result
        assert 'stats' in result

        tree = result['tree']
        assert tree is not None
        assert tree.get('role') is not None

        # Verify stats
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

        # Flatten the tree to find all roles
        roles = _collect_roles(tree)
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

    def test_server_serves_files(self, headless_driver: webdriver.Chrome, fixture_server: str) -> None:
        """Navigate to an HTML fixture served via HTTP."""
        # Create a simple test file
        test_file = Path(__file__).parent / 'test_error_handling.html'
        if test_file.exists():
            headless_driver.get(f'{fixture_server}/test_error_handling.html')
            assert headless_driver.title  # Page loaded successfully


# ============================================================================
# PROTOTYPE 2: YAML-driven test spec runner
# ============================================================================


def _collect_roles(node: dict[str, Any]) -> set[str]:
    """Recursively collect all roles from an ARIA tree."""
    roles: set[str] = set()
    if node is None:
        return roles
    if 'role' in node:
        roles.add(node['role'])
    for child in node.get('children', []):
        roles.update(_collect_roles(child))
    return roles


def _serialize_aria_tree(node: dict[str, Any] | None, indent: int = 0) -> str:
    """Minimal YAML serializer matching the MCP server's output format.

    This is a simplified version of the serializer in server.py.
    For full fidelity testing, we'd import and call the server's serializer.
    """
    if node is None:
        return ''

    lines = []
    prefix = ' ' * indent + '- '

    if node.get('type') == 'text':
        content = ' '.join(node.get('content', '').split())
        lines.append(f'{prefix}text: {content}')
        return '\n'.join(lines)

    role = node.get('role', 'generic')
    name = node.get('name', '')
    children = node.get('children', [])

    header = f'{prefix}{role}'
    if name:
        escaped_name = name.replace('"', '\\"')
        header += f' "{escaped_name}"'

    # Attributes
    attrs = []
    if 'level' in node:
        attrs.append(f'level={node["level"]}')
    if 'checked' in node:
        val = node['checked']
        if val == 'mixed':
            attrs.append('checked=mixed')
        elif val:
            attrs.append('checked')
        else:
            attrs.append('unchecked')
    if 'selected' in node:
        attrs.append('selected' if node['selected'] else 'selected=false')
    if 'pressed' in node:
        val = node['pressed']
        if val == 'mixed':
            attrs.append('pressed=mixed')
        elif val:
            attrs.append('pressed')
        else:
            attrs.append('pressed=false')
    if 'expanded' in node:
        attrs.append('expanded' if node['expanded'] else 'expanded=false')
    if node.get('disabled'):
        attrs.append('disabled')
    if node.get('url'):
        attrs.append(f'url={node["url"]}')
    if node.get('hidden'):
        attrs.append(f'hidden:{node["hidden"]}')
    if node.get('visuallyHidden'):
        attrs.append(f'visually-hidden:{node["visuallyHidden"]}')

    if attrs:
        header += f' [{", ".join(attrs)}]'
    if children:
        header += ':'

    lines.append(header)

    for child in children:
        child_output = _serialize_aria_tree(child, indent + 2)
        if child_output:
            lines.append(child_output)

    return '\n'.join(lines)


def _load_yaml_test_specs(yaml_path: Path) -> list[dict[str, Any]]:
    """Load test specs from a YAML file and flatten into parametrizable cases.

    Returns a list of dicts, each with:
    - id: test identifier
    - name: test name
    - section: which section the test belongs to
    - spec: the full test specification dict
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cases: list[dict[str, Any]] = []

    for section_key, section in data.items():
        if section_key == 'metadata':
            continue
        if not isinstance(section, dict):
            continue

        # Some sections have direct tests, some have nested tests
        tests = section.get('tests', [])
        if not tests and 'selector' in section:
            # Section is itself a test case (like page_metadata_tests.yaml)
            cases.append(
                {
                    'id': section_key,
                    'name': section_key,
                    'section': section_key,
                    'spec': section,
                }
            )
        else:
            cases.extend(
                {
                    'id': test.get('id', 'unknown'),
                    'name': test.get('name', 'unnamed'),
                    'section': section_key,
                    'spec': test,
                }
                for test in tests
            )

    return cases


def assert_includes_excludes(output: str, includes: list[str], excludes: list[str]) -> None:
    """Assert that output contains all includes and none of excludes."""
    for expected in includes:
        assert expected in output, f'MISSING expected string: {expected!r}\n--- Output ---\n{output[:2000]}'
    for forbidden in excludes:
        assert forbidden not in output, f'FOUND forbidden string: {forbidden!r}\n--- Output ---\n{output[:2000]}'


# Load the include_hidden test specs for parametrization
_INCLUDE_HIDDEN_YAML = Path(__file__).parent / 'include_hidden_tests.yaml'
_include_hidden_specs: list[dict[str, Any]] = []
if _INCLUDE_HIDDEN_YAML.exists():
    _include_hidden_specs = _load_yaml_test_specs(_INCLUDE_HIDDEN_YAML)


class TestYamlSpecParsing:
    """Verify the YAML spec loading and assertion framework works."""

    def test_load_include_hidden_specs(self) -> None:
        """Verify we can parse the include_hidden_tests.yaml file."""
        assert len(_include_hidden_specs) > 0, 'No test specs found'

        # Verify structure
        for case in _include_hidden_specs:
            assert 'id' in case
            assert 'name' in case
            assert 'spec' in case

    def test_load_compact_tree_specs(self) -> None:
        """Verify we can parse the compact_tree_tests.yaml file."""
        yaml_path = Path(__file__).parent / 'compact_tree_tests.yaml'
        specs = _load_yaml_test_specs(yaml_path)
        assert len(specs) > 0

        # Verify a specific test case
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


class TestSerializerFidelity:
    """Verify our test serializer matches the MCP server's output format."""

    def test_simple_tree_serialization(self) -> None:
        tree = {
            'role': 'button',
            'name': 'Submit',
        }
        output = _serialize_aria_tree(tree)
        assert output == '- button "Submit"'

    def test_heading_with_level(self) -> None:
        tree = {
            'role': 'heading',
            'name': 'Title',
            'level': 1,
        }
        output = _serialize_aria_tree(tree)
        assert output == '- heading "Title" [level=1]'

    def test_nested_tree(self) -> None:
        tree = {
            'role': 'navigation',
            'name': 'Main',
            'children': [
                {'role': 'link', 'name': 'Home'},
                {'role': 'link', 'name': 'About'},
            ],
        }
        output = _serialize_aria_tree(tree)
        assert '- navigation "Main":' in output
        assert '  - link "Home"' in output
        assert '  - link "About"' in output

    def test_text_node(self) -> None:
        tree = {
            'type': 'text',
            'content': 'Hello world',
        }
        output = _serialize_aria_tree(tree)
        assert output == '- text: Hello world'

    def test_hidden_marker(self) -> None:
        tree = {
            'role': 'button',
            'name': 'Hidden',
            'hidden': 'aria-hidden',
        }
        output = _serialize_aria_tree(tree)
        assert '[hidden:aria-hidden]' in output

    def test_visually_hidden_marker(self) -> None:
        tree = {
            'role': 'generic',
            'name': 'SR Only',
            'visuallyHidden': 'clipped',
        }
        output = _serialize_aria_tree(tree)
        assert '[visually-hidden:clipped]' in output


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
        output = _serialize_aria_tree(tree)

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
        output = _serialize_aria_tree(result['tree'])

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

        # include_hidden=False (default)
        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, False)
        output = _serialize_aria_tree(result['tree'])

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

        # include_hidden=True
        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, True)
        output = _serialize_aria_tree(result['tree'])

        assert 'button "Visible"' in output
        assert 'Hidden' in output
        assert '[hidden:aria-hidden]' in output
