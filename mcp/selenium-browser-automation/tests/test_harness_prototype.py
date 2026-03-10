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
from selenium_browser_automation.server import (
    _ARIA_HIDDEN_REASON_KEYS,
    _VISUAL_HIDDEN_REASON_KEYS,
    _build_page_metadata,
    _count_tree_nodes,
)
from selenium_browser_automation.tree_utils import (
    compact_aria_tree,
    compact_visual_tree,
    serialize_aria_snapshot,
    serialize_visual_tree,
)

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


@pytest.fixture(scope='session')
def examples_server() -> Any:
    """Session-scoped HTTP server for examples/ directory.

    Serves from repo root to support both examples/ and tests/ paths.
    """
    repo_root = Path(__file__).parent.parent

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        """HTTP handler that suppresses access logs."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(repo_root), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            pass

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

    # Extract global defaults from metadata
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

        # Some sections have direct tests, some have nested tests
        tests = section.get('tests', [])
        if not tests and ('selector' in section or 'tool' in section):
            # Section is itself a test case (like page_metadata_tests.yaml)
            cases.append(
                {
                    'id': section_key,
                    'name': section_key,
                    'section': section_key,
                    'spec': {**global_defaults, **section},
                }
            )
        else:
            # Merge section-level fields (selector, fixture) into each test
            section_defaults = {k: v for k, v in section.items() if k not in ['tests', 'description']}
            cases.extend(
                {
                    'id': test.get('id', 'unknown'),
                    'name': test.get('name', 'unnamed'),
                    'section': section_key,
                    'spec': {**global_defaults, **section_defaults, **test},  # Globals, section, test
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
        output = serialize_aria_snapshot(tree)
        assert output == '- button "Submit"'

    def test_heading_with_level(self) -> None:
        tree = {
            'role': 'heading',
            'name': 'Title',
            'level': 1,
        }
        output = serialize_aria_snapshot(tree)
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
        output = serialize_aria_snapshot(tree)
        assert '- navigation "Main":' in output
        assert '  - link "Home"' in output
        assert '  - link "About"' in output

    def test_text_node(self) -> None:
        tree = {
            'type': 'text',
            'content': 'Hello world',
        }
        output = serialize_aria_snapshot(tree)
        assert output == '- text: Hello world'

    def test_hidden_marker(self) -> None:
        tree = {
            'role': 'button',
            'name': 'Hidden',
            'hidden': 'aria-hidden',
        }
        output = serialize_aria_snapshot(tree)
        assert '[hidden:aria-hidden]' in output

    def test_visually_hidden_marker(self) -> None:
        tree = {
            'role': 'generic',
            'name': 'SR Only',
            'visuallyHidden': 'clipped',
        }
        output = serialize_aria_snapshot(tree)
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

        # include_hidden=False (default)
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

        # include_hidden=True
        result = headless_driver.execute_script(ARIA_SNAPSHOT_SCRIPT, 'body', False, True)
        output = serialize_aria_snapshot(result['tree'])

        assert 'button "Visible"' in output
        assert 'Hidden' in output
        assert '[hidden:aria-hidden]' in output


# ============================================================================
# YAML-DRIVEN INTEGRATION TESTS
# ============================================================================


def _extract_assertions(spec: dict[str, Any], mode: str = 'tree') -> tuple[list[str], list[str]]:
    """Extract includes/excludes from various YAML assertion formats.

    Args:
        spec: Test specification dict
        mode: 'tree' for tree assertions, 'metadata' for metadata assertions

    Returns:
        (includes, excludes) tuple of assertion strings
    """
    # Try different assertion keys in order
    for key in ['expect', 'compact', 'full', 'metadata', 'tree']:
        if key in spec:
            # For page_metadata tests with separate tree/metadata assertions
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


def _execute_tree_pipeline(
    driver: webdriver.Chrome,
    fixture_url: str,
    selector: str,
    tool_name: str,  # 'aria' or 'visual'
    include_urls: bool,
    include_hidden: bool,
    compact_tree: bool,
    include_page_info: bool,
) -> tuple[str, str]:
    """Execute the full tree snapshot pipeline, returning (tree_yaml, metadata_footer).

    Extracts the core logic so callers can assert on tree body and metadata footer
    independently (needed for page_metadata tests with separate assertion zones).
    """
    # Navigate to fixture
    driver.get(fixture_url)

    # Execute appropriate JS script
    if tool_name == 'aria':
        script = ARIA_SNAPSHOT_SCRIPT
        reason_keys = _ARIA_HIDDEN_REASON_KEYS
    else:
        script = VISUAL_TREE_SCRIPT
        reason_keys = _VISUAL_HIDDEN_REASON_KEYS

    result = driver.execute_script(script, selector, include_urls, include_hidden)

    # Destructure result
    if result is None:
        tree_data = None
        stats: dict[str, Any] = {}
    else:
        tree_data = result.get('tree')
        stats = result.get('stats', {})

    # Count nodes before compaction
    raw_count = _count_tree_nodes(tree_data)

    # Apply Python compaction if requested
    if compact_tree and tree_data:
        if tool_name == 'aria':
            tree_data = compact_aria_tree(tree_data)
        else:
            tree_data = compact_visual_tree(tree_data)

    # Serialize to YAML
    if tool_name == 'aria':
        tree_yaml = serialize_aria_snapshot(tree_data)
    else:
        tree_yaml = serialize_visual_tree(tree_data)

    # Build page metadata footer
    compacted_count = _count_tree_nodes(tree_data)
    metadata_footer = _build_page_metadata(
        page_stats=stats,
        include_page_info=include_page_info,
        include_urls=include_urls,
        compact_tree=compact_tree,
        raw_node_count=raw_count,
        compacted_node_count=compacted_count,
        hidden_reason_keys=reason_keys,
    )

    return tree_yaml, metadata_footer or ''


def _run_tree_test(
    driver: webdriver.Chrome,
    fixture_url: str,
    selector: str,
    tool_name: str,  # 'aria' or 'visual'
    include_urls: bool,
    include_hidden: bool,
    compact_tree: bool,
    include_page_info: bool,
    includes: list[str],
    excludes: list[str],
) -> None:
    """Execute a tree snapshot test and assert includes/excludes on combined output."""
    tree_yaml, metadata_footer = _execute_tree_pipeline(
        driver, fixture_url, selector, tool_name, include_urls, include_hidden, compact_tree, include_page_info
    )
    combined = tree_yaml + metadata_footer
    assert_includes_excludes(combined, includes, excludes)


# ============================================================================
# YAML-DRIVEN PARAMETRIZED TESTS
# ============================================================================


@pytest.mark.parametrize(
    'test_case',
    _load_yaml_test_specs(Path(__file__).parent / 'compact_tree_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_compact_tree_spec(headless_driver: webdriver.Chrome, examples_server: str, test_case: dict[str, Any]) -> None:
    """Run all compact_tree_tests.yaml test cases."""
    spec = test_case['spec']

    # Extract test parameters
    selector = spec.get('selector', 'body')
    fixture_path = spec.get('fixture', 'examples/compact-tree.html')
    fixture_url = f'{examples_server}/{fixture_path}'

    # Test with compact_tree=True
    if 'compact' in spec:
        includes, excludes = _extract_assertions(
            spec['compact'] if isinstance(spec.get('compact'), dict) else spec, 'tree'
        )
        _run_tree_test(
            driver=headless_driver,
            fixture_url=fixture_url,
            selector=selector,
            tool_name='aria',
            include_urls=False,
            include_hidden=False,
            compact_tree=True,
            include_page_info=False,
            includes=includes,
            excludes=excludes,
        )

    # Test with compact_tree=False (full output)
    if 'full' in spec:
        includes, excludes = _extract_assertions(spec['full'] if isinstance(spec.get('full'), dict) else spec, 'tree')
        _run_tree_test(
            driver=headless_driver,
            fixture_url=fixture_url,
            selector=selector,
            tool_name='aria',
            include_urls=False,
            include_hidden=False,
            compact_tree=False,
            include_page_info=False,
            includes=includes,
            excludes=excludes,
        )


@pytest.mark.parametrize(
    'test_case',
    _load_yaml_test_specs(Path(__file__).parent / 'include_hidden_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_include_hidden_spec(
    headless_driver: webdriver.Chrome, examples_server: str, test_case: dict[str, Any]
) -> None:
    """Run all include_hidden_tests.yaml test cases."""
    spec = test_case['spec']

    # Extract test parameters
    selector = spec.get('selector', 'body')
    include_hidden = spec.get('include_hidden', False)
    fixture_path = spec.get('fixture', 'examples/include-hidden-test.html')
    fixture_url = f'{examples_server}/{fixture_path}'

    # Extract assertions
    includes, excludes = _extract_assertions(spec, 'tree')

    _run_tree_test(
        driver=headless_driver,
        fixture_url=fixture_url,
        selector=selector,
        tool_name='aria',
        include_urls=False,
        include_hidden=include_hidden,
        compact_tree=False,
        include_page_info=False,
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    _load_yaml_test_specs(Path(__file__).parent / 'visual_tree_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_visual_tree_spec(headless_driver: webdriver.Chrome, examples_server: str, test_case: dict[str, Any]) -> None:
    """Run all visual_tree_tests.yaml test cases."""
    spec = test_case['spec']

    # Extract test parameters
    selector = spec.get('selector', 'body')
    include_hidden = spec.get('include_hidden', False)
    compact_tree = spec.get('compact_tree', False)
    fixture_path = spec.get('fixture', 'examples/visual-tree-test.html')
    fixture_url = f'{examples_server}/{fixture_path}'

    # Extract assertions
    includes, excludes = _extract_assertions(spec, 'tree')

    _run_tree_test(
        driver=headless_driver,
        fixture_url=fixture_url,
        selector=selector,
        tool_name='visual',
        include_urls=False,
        include_hidden=include_hidden,
        compact_tree=compact_tree,
        include_page_info=False,
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    _load_yaml_test_specs(Path(__file__).parent / 'state_detection_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_state_detection_spec(
    headless_driver: webdriver.Chrome, examples_server: str, test_case: dict[str, Any]
) -> None:
    """Run all state_detection_tests.yaml test cases."""
    spec = test_case['spec']

    # Extract test parameters
    selector = spec.get('selector', 'body')
    include_hidden = spec.get('include_hidden', False)
    compact_tree = spec.get('compact_tree', False)
    fixture_path = spec.get('fixture', 'examples/state-detection.html')
    fixture_url = f'{examples_server}/{fixture_path}'

    # Extract assertions
    includes, excludes = _extract_assertions(spec, 'tree')

    _run_tree_test(
        driver=headless_driver,
        fixture_url=fixture_url,
        selector=selector,
        tool_name='aria',  # State detection tests use ARIA snapshot
        include_urls=False,
        include_hidden=include_hidden,
        compact_tree=compact_tree,
        include_page_info=False,
        includes=includes,
        excludes=excludes,
    )


@pytest.mark.parametrize(
    'test_case',
    _load_yaml_test_specs(Path(__file__).parent / 'page_metadata_tests.yaml'),
    ids=lambda tc: tc['id'],
)
def test_page_metadata_spec(headless_driver: webdriver.Chrome, examples_server: str, test_case: dict[str, Any]) -> None:
    """Run all page_metadata_tests.yaml scenarios.

    These tests verify the two-tier metadata footer system by asserting
    tree body and metadata footer independently.
    """
    spec = test_case['spec']

    # Extract tool type
    tool_field = spec.get('tool', 'get_aria_snapshot')
    tool_name = 'visual' if 'visual' in tool_field.lower() else 'aria'

    # Extract args dict
    args = spec.get('args', {})
    selector = args.get('selector', 'body')
    include_page_info = args.get('include_page_info', False)
    compact_tree = args.get('compact_tree', True)
    include_urls = args.get('include_urls', False)
    include_hidden = args.get('include_hidden', False)

    # Build fixture URL
    fixture_path = spec.get('fixture', 'tests/test_page_metadata_integration.html')
    fixture_url = f'{examples_server}/{fixture_path}'

    # Execute pipeline
    tree_yaml, metadata_footer = _execute_tree_pipeline(
        driver=headless_driver,
        fixture_url=fixture_url,
        selector=selector,
        tool_name=tool_name,
        include_urls=include_urls,
        include_hidden=include_hidden,
        compact_tree=compact_tree,
        include_page_info=include_page_info,
    )

    # Assert metadata footer (separate zone)
    meta_includes, meta_excludes = _extract_assertions(spec, 'metadata')
    if meta_includes or meta_excludes:
        assert_includes_excludes(metadata_footer, meta_includes, meta_excludes)

    # Assert tree body (separate zone)
    tree_includes, tree_excludes = _extract_assertions(spec, 'tree')
    if tree_includes or tree_excludes:
        assert_includes_excludes(tree_yaml, tree_includes, tree_excludes)
