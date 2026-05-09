#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
"""Auth recipe — drives github.com login and hands off to gh-upload auth-import.

Resolves the GitHub credential item in 1Password by URL + exact title + canonical
login, runs the selenium-browser pipeline (16 steps including 2FA via authenticator
app), validates the captured profile state, and ingests it via
``gh-upload auth-import``. Step-by-step progress prints to stderr without exposing
credential values; pass an output state path as the first argument to capture the
saved state to a known file (default: a private tempfile).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.profile_state import ProfileState
from cc_lib.types import JsonObject

MATCH_URL = 'https://github.com'
MATCH_TITLE = 'GitHub'  # exact, case-sensitive
MATCH_LOGIN = 'ctguil@ucla.edu'  # canonical account identity (the email logged into github.com)

boundary = ErrorBoundary(exit_code=1)


@boundary
def main() -> None:
    """Run the auth recipe; exit 0 on success, non-zero on any failure."""
    state_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkstemp(suffix='.json')[1])

    creds = _fetch_creds(MATCH_URL, MATCH_TITLE, MATCH_LOGIN)
    pipeline = _build_pipeline(creds, state_path)

    _step('driving selenium pipeline (16 steps including 2FA)')
    subprocess.run(
        ['selenium-browser', 'pipeline', '-f', 'json'],
        input=json.dumps(pipeline).encode(),
        check=True,
    )

    _step('validating saved profile state')
    state = ProfileState.model_validate_json(state_path.read_text())
    if not state.cookies:
        raise AuthRecipeError('saved state has no cookies — login likely did not complete')

    _step(f'handing off to gh-upload auth-import (state={state_path})')
    subprocess.run(['gh-upload', 'auth-import', str(state_path)], check=True)


# -- Helpers (private) -------------------------------------------------------


def _step(msg: str) -> None:
    """Emit a labeled step header to stderr; never includes credential values."""
    print(f'→ {msg}', file=sys.stderr)


def _run(cmd: Sequence[str]) -> bytes:
    """Run a subprocess capturing stdout; raises CalledProcessError on non-zero exit."""
    return subprocess.run(list(cmd), capture_output=True, check=True).stdout


def _fetch_creds(match_url: str, match_title: str, match_login: str) -> Mapping[str, str]:
    """Resolve a 1Password item by URL + exact title + canonical login; return username, password, TOTP.

    Filter is URL-contains AND title-equals AND additional_information-equals. The
    ``additional_information`` field on a LOGIN item is the canonical account
    identity (visible without ``--reveal``, no extra biometric per candidate).
    """
    _step(f'looking up 1Password item: title={match_title!r}, login={match_login!r}, url contains {match_url!r}')
    items = json.loads(_run(['op', 'item', 'list', '--categories', 'login', '--format', 'json']))
    url_matches = [i for i in items if any(match_url in u.get('href', '') for u in i.get('urls', []))]
    matches = [
        i for i in url_matches if i.get('title') == match_title and i.get('additional_information') == match_login
    ]
    if not matches:
        candidates = [
            f'title={i.get("title")!r}, login={i.get("additional_information")!r}, id={i["id"]}' for i in url_matches
        ]
        raise AuthRecipeError(
            f'no 1Password item with title=={match_title!r} AND login=={match_login!r} among URL matches for '
            f'{match_url!r}; URL candidates: {candidates or "(none — no URL matches either)"}'
        )
    if len(matches) > 1:
        ids = [m['id'] for m in matches]
        raise AuthRecipeError(
            f'{len(matches)} items match title=={match_title!r} AND login=={match_login!r} (ids: {ids}); '
            f'refine MATCH_TITLE/MATCH_LOGIN in the recipe to disambiguate'
        )
    item_id = matches[0]['id']

    _step('fetching item fields (one biometric prompt)')
    detail = json.loads(_run(['op', 'item', 'get', item_id, '--reveal', '--format', 'json']))
    username = next((f.get('value') for f in detail['fields'] if f.get('purpose') == 'USERNAME'), None)
    password = next((f.get('value') for f in detail['fields'] if f.get('purpose') == 'PASSWORD'), None)
    if not username or not password:
        raise AuthRecipeError(f'1Password item {item_id} is missing USERNAME or PASSWORD field')

    _step('fetching fresh TOTP (rotates every 30s)')
    totp = _run(['op', 'item', 'get', item_id, '--otp']).decode().strip()

    return {'username': username, 'password': password, 'totp': totp}


def _build_pipeline(creds: Mapping[str, str], state_path: Path) -> Sequence[JsonObject]:
    """Build the selenium-browser pipeline JSON with credentials substituted in-process."""
    return [
        {
            'tool': 'navigate',
            'params': {'url': 'https://github.com/login', 'fresh_browser': True, 'browser': 'chromium'},
        },
        {'tool': 'wait_for_selector', 'params': {'css_selector': 'input.js-login-field'}},
        {'tool': 'click', 'params': {'css_selector': 'input.js-login-field'}},
        {'tool': 'type_text', 'params': {'text': creds['username']}},
        {'tool': 'click', 'params': {'css_selector': 'input.js-password-field'}},
        {'tool': 'type_text', 'params': {'text': creds['password']}},
        {'tool': 'click', 'params': {'css_selector': 'input.js-sign-in-button'}},
        {'tool': 'wait_for_network_idle', 'params': {'timeout': 10000}},
        {'tool': 'click', 'params': {'css_selector': 'button.more-options-two-factor'}},
        {'tool': 'wait_for_selector', 'params': {'css_selector': 'a[href*="two_factor_app_prompt"]'}},
        {'tool': 'click', 'params': {'css_selector': 'a[href*="two_factor_app_prompt"]'}},
        {'tool': 'wait_for_selector', 'params': {'css_selector': 'input.app_totp'}},
        {'tool': 'click', 'params': {'css_selector': 'input.app_totp'}},
        {'tool': 'type_text', 'params': {'text': creds['totp']}},
        {'tool': 'wait_for_network_idle', 'params': {'timeout': 10000}},
        {'tool': 'save_profile_state', 'params': {'filename': str(state_path)}},
    ]


# -- Exceptions + error boundary handlers ------------------------------------


class AuthRecipeError(Exception):
    """Recipe could not complete the authentication flow."""


@boundary.handler(subprocess.CalledProcessError)
def _handle_subprocess_failure(exc: subprocess.CalledProcessError) -> None:
    """Re-emit captured stderr (silenced by capture_output) and a one-line summary."""
    if exc.stderr:
        sys.stderr.buffer.write(exc.stderr)
    print(f'gh-upload-auth-recipe: {exc.cmd[0]} failed (exit {exc.returncode})', file=sys.stderr)


@boundary.handler(AuthRecipeError)
def _handle_recipe_error(exc: AuthRecipeError) -> None:
    print(f'gh-upload-auth-recipe: {exc}', file=sys.stderr)


@boundary.handler(Exception)
def _handle_unexpected(exc: Exception) -> None:
    print(f'gh-upload-auth-recipe: {type(exc).__name__}: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
