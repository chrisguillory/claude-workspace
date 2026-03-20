#!/usr/bin/env -S uv run --no-project
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "httpx",
#   "pydantic>=2.0.0",
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""Upload local files to GitHub issues/PRs using saved browser session cookies.

GitHub's file upload endpoint requires web session auth — gh CLI OAuth tokens
don't work. This script uses browser cookies saved via Selenium MCP profile state.

Usage:
    gh_upload.py <file_path> [<file_path2> ...]
    # Outputs markdown links: [file.csv](https://github.com/user-attachments/files/...)

Prerequisites:
    1. Navigate to github.com in Chromium via Selenium MCP with profile state
    2. save_profile_state("~/.claude-workspace/scripts/gh_upload_session.json")
    3. Session cookies last ~2 weeks. Re-save if uploads start failing (422).
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx
from cc_lib.error_boundary import ErrorBoundary
from cc_lib.types import JsonDatetime
from pydantic import BaseModel

# Hardcoded for personal use with mainstay-io/monorepo
REPO_SLUG = 'mainstay-io/monorepo'
REPO_ID = '839989396'
SESSION_FILE = Path.home() / '.claude-workspace' / 'scripts' / 'gh_upload_session.json'

# MIME type mapping for common file extensions
MIME_TYPES: Mapping[str, str] = {
    '.csv': 'text/csv',
    '.json': 'application/json',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
}
DEFAULT_MIME_TYPE = 'application/octet-stream'


class UploadError(Exception):
    """File upload to GitHub failed."""


class SessionExpiredError(UploadError):
    """GitHub session cookies are expired or invalid."""


class BrowserCookie(BaseModel):
    """Cookie from saved browser profile state."""

    name: str
    value: str
    domain: str = ''


class BrowserState(BaseModel):
    """Saved browser session state from Selenium save_profile_state."""

    cookies: Sequence[BrowserCookie]
    saved_at: JsonDatetime | None = None


class UploadAsset(BaseModel):
    """GitHub upload asset metadata."""

    href: str


class UploadPolicy(BaseModel):
    """GitHub upload policy response."""

    upload_url: str
    form: Mapping[str, str]
    asset_upload_url: str
    upload_authenticity_token: str
    asset: UploadAsset


boundary = ErrorBoundary(exit_code=1)


def main() -> None:
    """Entry point: validate file paths and upload."""
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <file> [<file2> ...]', file=sys.stderr)
        sys.exit(1)

    # Validate all file paths up front (fail-fast)
    file_paths = [Path(arg) for arg in sys.argv[1:]]
    for path in file_paths:
        if not path.is_file():
            print(f'ERROR: {path} not found or not a regular file', file=sys.stderr)
            sys.exit(1)

    upload_files(file_paths)


@boundary
def upload_files(file_paths: Sequence[Path]) -> None:
    """Load session and upload files. ErrorBoundary catches exceptions."""
    session = load_session()
    nonce = get_fresh_nonce(session)

    for file_path in file_paths:
        url = upload_file(session, nonce, file_path)
        print(f'[{file_path.name}]({url})')


def load_session() -> httpx.Client:
    """Load GitHub session from saved browser profile state.

    GitHub's /upload/policies/assets endpoint requires:
      - Session cookies (not OAuth tokens — web endpoint, not API)
      - Browser-like headers (Origin, sec-fetch-site, sec-fetch-mode, sec-fetch-dest)
      - X-Fetch-Nonce CSRF token (fetched per-session via get_fresh_nonce)

    Without these headers → 422. The endpoint validates requests came from a browser.

    Raises FileNotFoundError if session file doesn't exist, ValidationError if malformed.
    """
    state = BrowserState.model_validate_json(SESSION_FILE.read_text())

    github_cookies = {c.name: c.value for c in state.cookies if c.domain in ('github.com', '.github.com')}

    return httpx.Client(
        cookies=github_cookies,
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


def get_fresh_nonce(client: httpx.Client) -> str:
    """Fetch issues/new page to extract CSRF nonce tied to this session.

    The X-Fetch-Nonce is cryptographically bound to the session cookies. Each
    session needs a fresh nonce — reusing a stale one or one from a different
    session → 422.

    Raises SessionExpiredError if redirected to login (cookies expired).
    Raises UploadError if nonce cannot be extracted from page.
    """
    resp = client.get(
        f'https://github.com/{REPO_SLUG}/issues/new',
        headers={
            'Accept': 'text/html',
            'sec-fetch-site': 'none',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-dest': 'document',
        },
    )
    print(f'DEBUG: Page fetch status={resp.status_code}, length={len(resp.text)}', file=sys.stderr)

    nonce_match = re.search(r'name="fetch-nonce"\s+content="([^"]+)"', resp.text)
    if not nonce_match:
        if 'login' in str(resp.url):
            raise SessionExpiredError('Session expired — redirected to login page')
        raise UploadError(f'Could not extract fetch-nonce from {resp.url}')

    nonce = nonce_match.group(1)
    print(f'DEBUG: Got nonce={nonce[:20]}...', file=sys.stderr)
    return nonce


def upload_file(client: httpx.Client, nonce: str, file_path: Path) -> str:
    """Upload file to GitHub via 3-step flow.

    GitHub's file upload flow (captured via HAR from browser):
      1. POST /upload/policies/assets with file metadata → returns S3 signed URL,
         form fields for S3 upload, and authenticity token (201 expected)
      2. POST file to S3 signed URL using form fields from step 1 → 204 expected
      3. PUT /upload/repository-files/{asset_id} with authenticity token to confirm
         → 200 expected, returns permanent github.com URL

    All three steps must succeed. A 422 on step 1 indicates expired session.

    Returns permanent GitHub URL for the uploaded file.
    Raises UploadError for any step failure, SessionExpiredError if session invalid.
    """
    file_name = file_path.name
    file_size = file_path.stat().st_size
    mime = MIME_TYPES.get(file_path.suffix.lower(), DEFAULT_MIME_TYPE)

    # Step 1: Get upload policy from GitHub
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

    # Step 2: Upload to S3 (bare httpx.post — S3 shouldn't receive GitHub cookies/headers)
    with file_path.open('rb') as f:
        files = {'file': (file_name, f, mime)}
        s3_resp = httpx.post(policy.upload_url, data=policy.form, files=files)

    if s3_resp.status_code != 204:
        raise UploadError(f'S3 upload failed ({s3_resp.status_code})')

    # Step 3: Confirm upload with GitHub
    confirm_resp = client.put(
        f'https://github.com{policy.asset_upload_url}',
        data={'authenticity_token': policy.upload_authenticity_token},
        headers={'X-Fetch-Nonce': nonce},
    )

    if confirm_resp.status_code != 200:
        raise UploadError(f'Confirm failed ({confirm_resp.status_code})')

    return policy.asset.href


# Exception handlers — translate exceptions to user-friendly stderr messages


@boundary.handler(SessionExpiredError)
def _handle_session_expired(exc: SessionExpiredError) -> None:
    print(f'ERROR: {exc}', file=sys.stderr)
    print(f'Re-save browser session: save_profile_state("{SESSION_FILE}")', file=sys.stderr)


@boundary.handler(UploadError)
def _handle_upload_error(exc: UploadError) -> None:
    print(f'ERROR: {exc}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'ERROR: Unexpected failure: {exc!r}', file=sys.stderr)


if __name__ == '__main__':
    main()
