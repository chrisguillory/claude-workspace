"""CLI client for selenium-browser-automation MCP server.

Connects to the running MCP server's HTTP bridge via Unix domain socket.
Each command dispatches to a BrowserService method on the shared WebDriver.
"""

from __future__ import annotations

__all__ = [
    'app',
    'main',
]

import json
import logging
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import httpx
import typer
from cc_lib.cli import add_completion_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

logger = logging.getLogger(__name__)

app = create_app(help='Selenium browser automation CLI — operates on the running MCP server browser.')
add_completion_command(app)
error_boundary = ErrorBoundary(exit_code=1)


# -- Navigation commands --


@app.command('navigate', rich_help_panel='Navigation')
@error_boundary
def navigate(
    url: Annotated[str, typer.Argument(help='URL to navigate to.')],
    fresh: Annotated[bool, typer.Option('--fresh', help='Close and reopen browser.')] = False,
    browser: Annotated[str | None, typer.Option('--browser', '-b', help='chrome or chromium.')] = None,
    har: Annotated[bool, typer.Option('--har', help='Enable HAR capture (requires --fresh).')] = False,
    init_script: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str] | None, typer.Option('--init-script', help='JS to inject before page load (repeatable).')
    ] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Navigate to a URL."""
    result = _call_tool(
        'navigate', url=url, fresh_browser=fresh, browser=browser, enable_har_capture=har, init_scripts=init_script
    )
    _print_result(result, format)


# -- Interaction commands --


@app.command('click', rich_help_panel='Interaction')
@error_boundary
def click(
    selector: Annotated[str, typer.Argument(help='CSS selector of element to click.')],
    wait_for_network: Annotated[bool, typer.Option('--wait-for-network', help='Add delay after click.')] = False,
    network_timeout: Annotated[int, typer.Option('--network-timeout', help='Delay duration in ms.')] = 10000,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Click an element."""
    result = _call_tool(
        'click', css_selector=selector, wait_for_network=wait_for_network, network_timeout=network_timeout
    )
    _print_result(result, format)


@app.command('type-text', rich_help_panel='Interaction')
@error_boundary
def type_text(
    text: Annotated[str, typer.Argument(help='Text to type.')],
    delay: Annotated[int, typer.Option('--delay', help='Delay between keystrokes (ms).')] = 0,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Type text character by character."""
    result = _call_tool('type_text', text=text, delay_ms=delay)
    _print_result(result, format)


@app.command('press-key', rich_help_panel='Interaction')
@error_boundary
def press_key(
    key: Annotated[str, typer.Argument(help='Key name (e.g., ENTER, TAB, ESCAPE).')],
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Press a keyboard key."""
    result = _call_tool('press_key', key=key)
    _print_result(result, format)


@app.command('hover', rich_help_panel='Interaction')
@error_boundary
def hover(
    selector: Annotated[str, typer.Argument(help='CSS selector of element to hover.')],
    duration: Annotated[int, typer.Option('--duration', help='Hover duration (ms).')] = 0,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Hover over an element."""
    result = _call_tool('hover', css_selector=selector, duration_ms=duration)
    _print_result(result, format)


@app.command('scroll', rich_help_panel='Interaction')
@error_boundary
def scroll(
    direction: Annotated[str | None, typer.Option('--direction', '-d', help='up, down, left, right.')] = None,
    amount: Annotated[int, typer.Option('--amount', '-n', help='Scroll amount.')] = 3,
    selector: Annotated[str | None, typer.Option('--selector', '-s', help='Element to scroll.')] = None,
    behavior: Annotated[Literal['instant', 'smooth'], typer.Option('--behavior', help='Scroll animation.')] = 'instant',
    position: Annotated[
        str | None, typer.Option('--position', '-p', help='Absolute: top, bottom, left, right.')
    ] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Scroll the page or an element."""
    result = _call_tool(
        'scroll', direction=direction, scroll_amount=amount, css_selector=selector, behavior=behavior, position=position
    )
    _print_result(result, format)


# -- Wait commands --


@app.command('wait-for-selector', rich_help_panel='Wait')
@error_boundary
def wait_for_selector(
    selector: Annotated[str, typer.Argument(help='CSS selector to wait for.')],
    state: Annotated[str, typer.Option('--state', help='visible, hidden, attached, detached.')] = 'visible',
    timeout: Annotated[int, typer.Option('--timeout', '-t', help='Timeout in ms.')] = 30000,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Wait for an element."""
    result = _call_tool('wait_for_selector', css_selector=selector, state=state, timeout=timeout)
    _print_result(result, format)


@app.command('wait-for-network-idle', rich_help_panel='Wait')
@error_boundary
def wait_for_network_idle(
    timeout: Annotated[int, typer.Option('--timeout', '-t', help='Timeout in ms.')] = 10000,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Wait for network to be idle."""
    result = _call_tool('wait_for_network_idle', timeout=timeout)
    _print_result(result, format)


# -- Capture commands --


@app.command('screenshot', rich_help_panel='Capture')
@error_boundary
def screenshot(
    filename: Annotated[str, typer.Argument(help='Output filename.')],
    full_page: Annotated[bool, typer.Option('--full-page', help='Capture full scrollable page.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Take a screenshot."""
    result = _call_tool('screenshot', filename=filename, full_page=full_page)
    _print_result(result, format)


@app.command('get-page-text', rich_help_panel='Content')
@error_boundary
def get_page_text(
    selector: Annotated[str, typer.Argument(help='CSS selector (default: auto).')] = 'auto',
    images: Annotated[bool, typer.Option('--images', help='Include image descriptions.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Extract page text."""
    result = _call_tool('get_page_text', selector=selector, include_images=images)
    _print_result(result, format)


@app.command('get-page-html', rich_help_panel='Content')
@error_boundary
def get_page_html(
    selector: Annotated[str | None, typer.Argument(help='CSS selector (default: full page).')] = None,
    limit: Annotated[int | None, typer.Option('--limit', '-n', help='Max elements to return.')] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get raw HTML source."""
    result = _call_tool('get_page_html', selector=selector, limit=limit)
    _print_result(result, format)


@app.command('get-interactive-elements', rich_help_panel='Content')
@error_boundary
def get_interactive_elements(
    scope: Annotated[str, typer.Argument(help='CSS selector scope (e.g., "body", ".wizard").')],
    text_contains: Annotated[str | None, typer.Option('--text', help='Filter by text content.')] = None,
    tag_filter: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str] | None, typer.Option('--tag', help='Filter by HTML tag (repeatable).')
    ] = None,
    limit: Annotated[int | None, typer.Option('--limit', '-n', help='Max results.')] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Find interactive elements by text or scope."""
    result = _call_tool(
        'get_interactive_elements',
        selector_scope=scope,
        text_contains=text_contains,
        tag_filter=tag_filter,
        limit=limit,
    )
    _print_result(result, format)


@app.command('get-aria-snapshot', rich_help_panel='Content')
@error_boundary
def get_aria_snapshot(
    selector: Annotated[str, typer.Argument(help='CSS selector scope.')] = 'body',
    urls: Annotated[bool, typer.Option('--urls', help='Include href values.')] = False,
    hidden: Annotated[bool, typer.Option('--hidden', help='Include hidden elements.')] = False,
    compact: Annotated[bool, typer.Option('--compact/--no-compact', help='Remove structural noise.')] = True,
    page_info: Annotated[bool, typer.Option('--page-info', help='Show extended page stats.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get ARIA accessibility snapshot."""
    result = _call_tool(
        'get_aria_snapshot',
        selector=selector,
        include_urls=urls,
        include_hidden=hidden,
        compact_tree=compact,
        include_page_info=page_info,
    )
    _print_result(result, format)


@app.command('execute-javascript', rich_help_panel='Capture')
@error_boundary
def execute_javascript(
    code: Annotated[str, typer.Argument(help='JavaScript code to execute.')],
    timeout: Annotated[int, typer.Option('--timeout', '-t', help='Timeout in ms.')] = 30000,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Execute JavaScript."""
    result = _call_tool('execute_javascript', code=code, timeout_ms=timeout)
    _print_result(result, format)


@app.command('capture-web-vitals', rich_help_panel='Capture')
@error_boundary
def capture_web_vitals(
    timeout: Annotated[int, typer.Option('--timeout', '-t', help='Timeout in ms.')] = 5000,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Capture Core Web Vitals."""
    result = _call_tool('capture_web_vitals', timeout_ms=timeout)
    _print_result(result, format)


@app.command('export-har', rich_help_panel='Capture')
@error_boundary
def export_har(
    filename: Annotated[str, typer.Argument(help='Output HAR filename.')],
    bodies: Annotated[bool, typer.Option('--bodies', help='Include response bodies.')] = False,
    max_body_size: Annotated[int, typer.Option('--max-body-size', help='Max response body size in MB.')] = 10,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Export HAR network capture."""
    result = _call_tool('export_har', filename=filename, include_response_bodies=bodies, max_body_size_mb=max_body_size)
    _print_result(result, format)


@app.command('navigate-with-profile-state', rich_help_panel='Navigation')
@error_boundary
def navigate_with_profile_state(
    url: Annotated[str, typer.Argument(help='URL to navigate to after profile state import.')],
    profile_state_file: Annotated[
        str | None, typer.Option('--profile-state-file', help='Path to ProfileState JSON.')
    ] = None,
    chrome_profile: Annotated[str | None, typer.Option('--chrome-profile', help='Chrome profile name.')] = None,
    origin: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str] | None, typer.Option('--origin', help='Filter origins (repeatable).')
    ] = None,
    live_session_storage: Annotated[
        bool,
        typer.Option(
            '--live-session-storage/--no-live-session-storage', help='Extract live sessionStorage via AppleScript.'
        ),
    ] = False,
    browser: Annotated[str | None, typer.Option('--browser', '-b', help='chrome or chromium.')] = None,
    har: Annotated[bool, typer.Option('--har', help='Enable HAR capture.')] = False,
    init_script: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str] | None, typer.Option('--init-script', help='JS to inject before page load (repeatable).')
    ] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Navigate with imported profile state (cookies, localStorage)."""
    result = _call_tool(
        'navigate_with_profile_state',
        url=url,
        profile_state_file=profile_state_file,
        chrome_profile=chrome_profile,
        origins_filter=origin,
        live_session_storage_via_applescript=live_session_storage,
        browser=browser,
        enable_har_capture=har,
        init_scripts=init_script,
    )
    _print_result(result, format)


@app.command('get-visual-tree', rich_help_panel='Content')
@error_boundary
def get_visual_tree(
    selector: Annotated[str, typer.Argument(help='CSS selector scope.')] = 'body',
    urls: Annotated[bool, typer.Option('--urls', help='Include href values.')] = False,
    hidden: Annotated[bool, typer.Option('--hidden', help='Include hidden elements.')] = False,
    compact: Annotated[bool, typer.Option('--compact/--no-compact', help='Remove structural noise.')] = True,
    page_info: Annotated[bool, typer.Option('--page-info', help='Show extended page stats.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get visual tree (what sighted users see)."""
    result = _call_tool(
        'get_visual_tree',
        selector=selector,
        include_urls=urls,
        include_hidden=hidden,
        compact_tree=compact,
        include_page_info=page_info,
    )
    _print_result(result, format)


@app.command('get-focusable-elements', rich_help_panel='Content')
@error_boundary
def get_focusable_elements(
    only_tabbable: Annotated[bool, typer.Option('--only-tabbable/--all', help='Tab-key-only vs all focusable.')] = True,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get keyboard-navigable elements."""
    result = _call_tool('get_focusable_elements', only_tabbable=only_tabbable)
    _print_result(result, format)


@app.command('download-resource', rich_help_panel='Content')
@error_boundary
def download_resource(
    url: Annotated[str, typer.Argument(help='URL to download (http/https/file).')],
    output_filename: Annotated[str, typer.Argument(help='Filename to save as.')],
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Download a resource using browser session cookies."""
    result = _call_tool('download_resource', url=url, output_filename=output_filename)
    _print_result(result, format)


@app.command('resize-window', rich_help_panel='Interaction')
@error_boundary
def resize_window(
    width: Annotated[int, typer.Argument(help='Window width in pixels.')],
    height: Annotated[int, typer.Argument(help='Window height in pixels.')],
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Resize the browser window."""
    result = _call_tool('resize_window', width=width, height=height)
    _print_result(result, format)


@app.command('sleep', rich_help_panel='Wait')
@error_boundary
def sleep_cmd(
    duration: Annotated[int, typer.Argument(help='Duration in milliseconds.')],
    reason: Annotated[str | None, typer.Option('--reason', help='Context for the delay.')] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Pause execution for a fixed duration."""
    result = _call_tool('sleep', duration_ms=duration, reason=reason)
    _print_result(result, format)


@app.command('get-resource-timings', rich_help_panel='Capture')
@error_boundary
def get_resource_timings(
    clear_buffer: Annotated[bool, typer.Option('--clear-buffer', help='Clear timing buffer after retrieval.')] = False,
    min_duration: Annotated[
        int, typer.Option('--min-duration', help='Only include requests slower than this (ms).')
    ] = 0,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get resource timing data from Performance API."""
    result = _call_tool('get_resource_timings', clear_resource_timing_buffer=clear_buffer, min_duration_ms=min_duration)
    _print_result(result, format)


@app.command('get-console-logs', rich_help_panel='Capture')
@error_boundary
def get_console_logs(
    level: Annotated[str | None, typer.Option('--level', help='Filter: ALL, SEVERE, WARNING, INFO.')] = None,
    pattern: Annotated[str | None, typer.Option('--pattern', help='Regex pattern to filter messages.')] = None,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Get browser console logs."""
    result = _call_tool('get_console_logs', level_filter=level, pattern=pattern)
    _print_result(result, format)


# -- Profile State commands --


@app.command('list-chrome-profiles', rich_help_panel='Profile State')
@error_boundary
def list_chrome_profiles(
    verbose: Annotated[bool, typer.Option('--verbose/--no-verbose', help='Include all metadata.')] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """List Chrome profiles with metadata."""
    result = _call_tool('list_chrome_profiles', verbose=verbose)
    _print_result(result, format)


@app.command('save-profile-state', rich_help_panel='Profile State')
@error_boundary
def save_profile_state(
    filename: Annotated[str, typer.Argument(help='Output filename (e.g., auth.json).')],
    include_indexeddb: Annotated[
        bool, typer.Option('--include-indexeddb/--no-include-indexeddb', help='Capture IndexedDB databases.')
    ] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Save browser storage state to JSON for session persistence."""
    result = _call_tool('save_profile_state', filename=filename, include_indexeddb=include_indexeddb)
    _print_result(result, format)


@app.command('export-chrome-profile-state', rich_help_panel='Profile State')
@error_boundary
def export_chrome_profile_state(
    output_file: Annotated[str, typer.Argument(help='Output JSON filename.')],
    chrome_profile: Annotated[str, typer.Option('--chrome-profile', help='Chrome profile name.')] = 'Default',
    include_session_storage: Annotated[
        bool, typer.Option('--include-session-storage/--no-include-session-storage', help='Include sessionStorage.')
    ] = True,
    include_indexeddb: Annotated[
        bool, typer.Option('--include-indexeddb/--no-include-indexeddb', help='Include IndexedDB.')
    ] = False,
    origin: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str] | None, typer.Option('--origin', help='Filter origins (repeatable).')
    ] = None,
    live_session_storage: Annotated[
        bool,
        typer.Option(
            '--live-session-storage/--no-live-session-storage', help='Extract live sessionStorage via AppleScript.'
        ),
    ] = False,
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Export profile state from Chrome's profile files."""
    result = _call_tool(
        'export_chrome_profile_state',
        output_file=output_file,
        chrome_profile=chrome_profile,
        include_session_storage=include_session_storage,
        include_indexeddb=include_indexeddb,
        origins_filter=origin,
        live_session_storage_via_applescript=live_session_storage,
    )
    _print_result(result, format)


# -- Proxy commands --


@app.command('configure-proxy', rich_help_panel='Proxy')
@error_boundary
def configure_proxy(
    host: Annotated[str, typer.Argument(help='Proxy host (e.g., brd.superproxy.io).')],
    port: Annotated[int, typer.Argument(help='Proxy port (e.g., 33335).')],
    username: Annotated[str, typer.Argument(help='Proxy username.')],
    password: Annotated[str, typer.Argument(help='Proxy password.')],
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Configure authenticated HTTP proxy via mitmproxy."""
    result = _call_tool('configure_proxy', host=host, port=port, username=username, password=password)
    _print_result(result, format)


@app.command('clear-proxy', rich_help_panel='Proxy')
@error_boundary
def clear_proxy(
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Clear proxy and return to direct connection."""
    result = _call_tool('clear_proxy')
    _print_result(result, format)


@app.command('set-blocked-urls', rich_help_panel='Proxy')
@error_boundary
def set_blocked_urls(
    url: Annotated[  # strict_typing_linter.py: mutable-type — typer requires list
        list[str], typer.Option('--url', help='URL pattern to block (repeatable). No flags = clear all.')
    ] = [],  # noqa: B006 — Typer creates fresh list per invocation
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Block network requests matching URL patterns."""
    result = _call_tool('set_blocked_urls', urls=url)
    _print_result(result, format)


# -- Batch command --


@app.command('pipeline', rich_help_panel='Batch')
@error_boundary
def pipeline(
    file: Annotated[str | None, typer.Option('--file', '-F', help='Read pipeline JSON from file.')] = None,
    on_error: Annotated[str, typer.Option('--on-error', help='stop or continue.')] = 'stop',
    format: Annotated[Literal['text', 'json'], typer.Option('--format', '-f', help='Output format.')] = 'text',
) -> None:
    """Execute a batch pipeline of tool calls.

    Accepts JSON from stdin or --file. Two formats supported:

    \b
    Wrapped:   {"steps": [{"tool": "click", "params": {...}}, ...]}
    Bare:      [{"tool": "click", "params": {...}}, ...]

    \b
    Examples:
        selenium-browser-automation pipeline < workflow.json
        selenium-browser-automation pipeline -F workflow.json
        selenium-browser-automation pipeline --on-error continue < steps.json
    """
    if file:
        data = json.loads(pathlib.Path(file).read_text())
    else:
        data = json.loads(sys.stdin.read())

    steps = data.get('steps', data) if isinstance(data, dict) else data
    result = _call_pipeline(steps, on_error=on_error)

    if format == 'json':
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    # Text format — per-step status
    if isinstance(result, dict):
        typer.secho(
            f'{result.get("status", "unknown")} ({result.get("completed", 0)}/{result.get("total", 0)} steps, {result.get("elapsed_ms", 0)}ms)',
            bold=True,
        )
        for step in result.get('results', []):
            status = step.get('status', '?')
            icon = '✓' if status == 'ok' else '✗' if status == 'error' else '⊘'
            color = (
                typer.colors.GREEN if status == 'ok' else typer.colors.RED if status == 'error' else typer.colors.YELLOW
            )
            typer.secho(f'  {icon} {step.get("tool", "?")} ({step.get("elapsed_ms", 0)}ms)', fg=color)
            if status == 'error' and step.get('error'):
                typer.echo(f'    {step["error"].get("message", "")}')


# -- Entry point --


def main() -> None:
    """Entry point for CLI."""
    run_app(app)


# -- Private helpers --


@app.callback(invoke_without_command=True)
def _configure_logging(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option('--verbose', '-v', help='Show detailed output')] = False,
) -> None:
    """Configure logging and show help when no command given."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stderr, force=True)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def _get_socket_path() -> pathlib.Path:
    """Find the Unix socket for the running MCP server's HTTP bridge.

    Requires CLAUDECODE=1 env (set by Claude's Bash tool on all children).
    Walks the process tree to find the Claude Code ancestor PID.
    """
    if not os.environ.get('CLAUDECODE'):
        typer.secho(
            'Error: selenium-browser-automation must be run inside Claude Code (CLAUDECODE env not set)',
            fg=typer.colors.RED,
            err=True,
        )
        raise SystemExit(1)

    current = os.getppid()
    for _ in range(20):
        result = subprocess.run(
            ['ps', '-p', str(current), '-o', 'ppid=,comm='],
            check=False,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            break
        parts = result.stdout.strip().split(None, 1)
        ppid = int(parts[0])
        comm = parts[1] if len(parts) > 1 else ''
        if 'claude' in comm.lower():
            sock = pathlib.Path(f'/tmp/selenium-browser-automation-{current}.sock')
            if not sock.exists():
                typer.secho(f'Error: Socket not found at {sock}', fg=typer.colors.RED, err=True)
                typer.echo('Is the selenium-browser-automation MCP server running?', err=True)
                raise SystemExit(1)
            return sock
        if ppid == 0:
            break
        current = ppid

    typer.secho('Error: Claude Code not found in process tree', fg=typer.colors.RED, err=True)
    raise SystemExit(1)


def _call_tool(tool: str, **params: object) -> object:
    """Call a single tool via the HTTP bridge."""
    socket_path = _get_socket_path()
    transport = httpx.HTTPTransport(uds=socket_path.as_posix())
    with httpx.Client(transport=transport, timeout=120.0) as client:
        response = client.post(
            'http://localhost/tool',
            json={'tool': tool, 'params': {k: v for k, v in params.items() if v is not None}},
        )
        response.raise_for_status()
        return response.json()


def _call_pipeline(steps: Sequence[Mapping[str, object]], on_error: str = 'stop') -> object:
    """Call the pipeline endpoint via the HTTP bridge."""
    socket_path = _get_socket_path()
    transport = httpx.HTTPTransport(uds=socket_path.as_posix())
    with httpx.Client(transport=transport, timeout=300.0) as client:
        response = client.post(
            'http://localhost/pipeline',
            json={'steps': steps, 'on_error': on_error},
        )
        response.raise_for_status()
        return response.json()


def _print_result(result: object, format: str) -> None:
    """Print tool result in requested format."""
    if format == 'json':
        typer.echo(json.dumps(result, indent=2, default=str))
    elif isinstance(result, dict):
        if result.get('status') == 'error':
            error = result.get('error', {})
            typer.secho(f'Error: {error.get("message", "Unknown error")}', fg=typer.colors.RED, err=True)
            raise SystemExit(1)
        # Text mode — print the result value
        val = result.get('result')
        if val is not None:
            if isinstance(val, str):
                typer.echo(val)
            else:
                typer.echo(json.dumps(val, indent=2, default=str))
        else:
            elapsed = result.get('elapsed_ms')
            msg = f'OK ({elapsed}ms)' if elapsed is not None else 'OK'
            typer.secho(msg, fg=typer.colors.GREEN)
    elif result is not None:
        typer.echo(str(result))
