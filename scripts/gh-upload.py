#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
#   "httpx",
#   "pydantic>=2.0.0",
#   "typer>=0.16.0",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""GitHub upload + session management via saved browser cookies.

GitHub's file-upload endpoint requires web-session auth (gh CLI OAuth tokens
don't work on /upload/policies/assets). This script consumes a profile-state
JSON captured by selenium-browser to drive that endpoint.
"""

from __future__ import annotations

__all__ = ['app', 'main']

import logging
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import httpx
import typer
from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.base import SubsetModel
from cc_lib.schemas.profile_state import ProfileState
from cc_lib.utils import get_claude_workspace_config_home_dir
from cc_lib.utils.atomic_write import atomic_write

# Hardcoded for personal use with mainstay-io/monorepo
REPO_SLUG = 'mainstay-io/monorepo'
REPO_ID = '839989396'
SESSION_FILE = get_claude_workspace_config_home_dir() / 'scripts' / 'gh_upload_session.json'

# GitHub session cookies required for /upload/policies/assets. Empirically:
# `user_session` is the primary auth cookie; `_gh_sess` carries CSRF state.
LOAD_BEARING_COOKIES: Sequence[str] = ('user_session', '_gh_sess')

MIME_TYPES: Mapping[str, str] = {
    '.csv': 'text/csv',
    '.html': 'text/html',
    '.json': 'application/json',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
}
DEFAULT_MIME_TYPE = 'application/octet-stream'


class UploadAsset(SubsetModel):
    """GitHub upload asset metadata."""

    href: str


class UploadPolicy(SubsetModel):
    """GitHub upload policy response."""

    upload_url: str
    form: Mapping[str, str]
    asset_upload_url: str
    upload_authenticity_token: str
    asset: UploadAsset


app = create_app(help='gh-upload — GitHub file uploads via saved browser cookies.')
boundary = ErrorBoundary(exit_code=1)


@app.callback(invoke_without_command=True)
def _configure_logging(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option('--verbose', '-v', help='Show detailed output.')] = False,
) -> None:
    """Configure logging and show help when no command given."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(message)s', stream=sys.stderr, force=True)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command('upload', rich_help_panel='Upload')
@boundary
def upload(
    files: Annotated[
        list[Path], typer.Argument(help='Files to upload.')
    ],  # strict_typing_linter.py: mutable-type — typer requires list
) -> None:
    """Upload one or more files to GitHub; print markdown links to stdout."""
    for path in files:
        if not path.is_file():
            typer.echo(f'ERROR: {path} not found or not a regular file', err=True)
            raise typer.Exit(1)
    client = _load_session()
    nonce = _fetch_nonce(client)
    for path in files:
        url = _upload_one(client, nonce, path)
        typer.echo(f'[{path.name}]({url})')


@app.command('auth-status', rich_help_panel='Auth')
@boundary
def auth_status(
    cookie_path: Annotated[Path, typer.Option('--cookie-path', help='Cookie file location.')] = SESSION_FILE,
) -> None:
    """Report cookie file presence and load-bearing-cookie coverage."""
    if not cookie_path.exists():
        typer.echo(f'No cookie file at {cookie_path}')
        typer.echo('Run the `gh-upload-auth` skill to bootstrap.')
        raise typer.Exit(2)
    state = ProfileState.model_validate_json(cookie_path.read_text())
    github = [c for c in state.cookies if c.domain in ('github.com', '.github.com')]
    present = {c.name for c in github}
    missing = tuple(name for name in LOAD_BEARING_COOKIES if name not in present)
    typer.echo(f'Cookie file: {cookie_path}')
    typer.echo(f'github.com cookies: {len(github)} (of {len(state.cookies)} total)')
    typer.echo(f'Load-bearing required: {", ".join(LOAD_BEARING_COOKIES)}')
    typer.echo(f'Missing: {", ".join(missing) if missing else "none ✓"}')


@app.command('auth-logout', rich_help_panel='Auth')
@boundary
def auth_logout(
    cookie_path: Annotated[Path, typer.Option('--cookie-path', help='Cookie file to delete.')] = SESSION_FILE,
) -> None:
    """Delete the cookie file."""
    if cookie_path.exists():
        cookie_path.unlink()
        typer.echo(f'Deleted {cookie_path}')
    else:
        typer.echo(f'No cookie file at {cookie_path} (nothing to do)')


@app.command('auth-import', rich_help_panel='Auth')
@boundary
def auth_import(
    state_file: Annotated[Path, typer.Argument(help='Profile-state JSON to import.')],
    cookie_path: Annotated[Path, typer.Option('--cookie-path', help='Where to write.')] = SESSION_FILE,
) -> None:
    """Import a profile-state JSON into gh-upload's cookie store."""
    state = ProfileState.model_validate_json(state_file.read_text())
    cookie_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write(cookie_path, state.model_dump_json().encode(), mode=0o600)
    github = [c for c in state.cookies if c.domain in ('github.com', '.github.com')]
    present = {c.name for c in github}
    missing = tuple(name for name in LOAD_BEARING_COOKIES if name not in present)
    typer.echo(f'Imported {len(state.cookies)} cookies → {cookie_path}')
    typer.echo(f'github.com cookies: {len(github)}')
    typer.echo(f'Missing load-bearing: {", ".join(missing) if missing else "none ✓"}')
    if missing:
        raise typer.Exit(2)


# -- Upload helpers (private) -------------------------------------------------


def _load_session() -> httpx.Client:
    """Build an authenticated httpx client from the saved profile state."""
    state = ProfileState.model_validate_json(SESSION_FILE.read_text())
    cookies = {c.name: c.value for c in state.cookies if c.domain in ('github.com', '.github.com')}
    return httpx.Client(
        cookies=cookies,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'GitHub-Verified-Fetch': 'true',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': f'https://github.com/{REPO_SLUG}/issues/new',
            'Origin': 'https://github.com',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'cors',
            'sec-fetch-dest': 'empty',
            'sec-ch-ua': '"Chromium";v="145", "Not:A-Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        },
        follow_redirects=True,
    )


def _fetch_nonce(client: httpx.Client) -> str:
    """Extract the per-session X-Fetch-Nonce from issues/new HTML."""
    resp = client.get(
        f'https://github.com/{REPO_SLUG}/issues/new',
        headers={
            'Accept': 'text/html',
            'sec-fetch-site': 'none',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-dest': 'document',
        },
    )
    nonce_match = re.search(r'name="fetch-nonce"\s+content="([^"]+)"', resp.text)
    if not nonce_match:
        if 'login' in str(resp.url):
            raise SessionExpiredError('Session expired — redirected to login page')
        raise UploadError(f'Could not extract fetch-nonce from {resp.url}')
    return nonce_match.group(1)


def _upload_one(client: httpx.Client, nonce: str, file_path: Path) -> str:
    """Run GitHub's 3-step upload flow for one file; return the permanent URL."""
    file_name = file_path.name
    file_size = file_path.stat().st_size
    mime = MIME_TYPES.get(file_path.suffix.lower(), DEFAULT_MIME_TYPE)

    # Step 1: get upload policy
    resp = client.post(
        'https://github.com/upload/policies/assets',
        data={
            'repository_id': REPO_ID,
            'name': file_name,
            'size': str(file_size),
            'content_type': mime,
        },
        headers={'X-Fetch-Nonce': nonce},
    )
    if resp.status_code == 422:
        raise SessionExpiredError(f'Policy request returned 422 — session likely expired: {resp.text[:200]}')
    if resp.status_code != 201:
        raise UploadError(f'Policy request failed ({resp.status_code}): {resp.text[:200]}')
    policy = UploadPolicy.model_validate(resp.json())

    # Step 2: upload to S3 (bare httpx — S3 must not see GitHub cookies)
    with file_path.open('rb') as f:
        s3_resp = httpx.post(policy.upload_url, data=policy.form, files={'file': (file_name, f, mime)})
    if s3_resp.status_code != 204:
        raise UploadError(f'S3 upload failed ({s3_resp.status_code})')

    # Step 3: confirm with GitHub
    confirm = client.put(
        f'https://github.com{policy.asset_upload_url}',
        data={'authenticity_token': policy.upload_authenticity_token},
        headers={'X-Fetch-Nonce': nonce},
    )
    if confirm.status_code != 200:
        raise UploadError(f'Confirm failed ({confirm.status_code})')
    return policy.asset.href


# Register documentation and shell-completion commands last so their panels
# appear after Upload and Auth in --help output.
add_help_command(app)
add_completion_command(app)


def main() -> None:
    """CLI entry point."""
    run_app(app)


# -- Exception types + handlers (file bottom per workspace convention) -------


class UploadError(Exception):
    """File upload to GitHub failed."""


class SessionExpiredError(UploadError):
    """GitHub session cookies are expired or invalid."""


@boundary.handler(SessionExpiredError)
def _handle_session_expired(exc: SessionExpiredError) -> None:
    typer.echo(f'ERROR: {exc}', err=True)
    typer.echo('Re-run the gh-upload-auth skill or `gh-upload auth-import <state.json>` to refresh.', err=True)


@boundary.handler(UploadError)
def _handle_upload_error(exc: UploadError) -> None:
    typer.echo(f'ERROR: {exc}', err=True)


if __name__ == '__main__':
    main()
