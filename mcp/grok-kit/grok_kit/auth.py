"""Cookie management for grok.com.

Reads the cookie file produced by the selenium-browser MCP's
``save_profile_state`` (snake_case JSON, see ``CookieRecord`` for the shape).
Formats the load-bearing cookies into a ``Cookie:`` header value for the
Speakeasy SDK. Detects expiration via cookie ``expires`` fields.

Bootstrap (re-extract cookies when expired) is currently external — run the
``selenium-browser`` MCP's ``save_profile_state`` against a logged-in
grok.com session, pointing at ``DEFAULT_COOKIE_PATH``. The CLI command
``grok-kit auth login`` will automate this in a follow-up.

Five cookies are load-bearing for grok.com's ``/rest/`` endpoints, verified
empirically by removing each in turn and observing 401:

- ``sso``, ``sso-rw``: X SSO session tokens (long-lived, ~1 year)
- ``x-userid``: User UUID (paired with sso)
- ``cf_clearance``: Cloudflare bot-management challenge response
  (12-72h lifetime, IP-bound)
- ``__cf_bm``: Cloudflare bot-management session (short, 30 min)

The Cloudflare cookies are the practical refresh driver — sso lasts a year,
but cf_clearance forces re-bootstrap most days.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from cc_lib.schemas.base import ClosedModel, SubsetModel

__all__ = [
    'DEFAULT_COOKIE_PATH',
    'LOAD_BEARING_COOKIES',
    'CookieRecord',
    'GrokCookies',
    'expired_load_bearing',
    'format_cookie_header',
    'load_cookies',
    'missing_load_bearing',
]


DEFAULT_COOKIE_PATH = Path.home() / '.config' / 'grok-kit' / 'cookies.json'

LOAD_BEARING_COOKIES: Sequence[str] = (
    'sso',
    'sso-rw',
    'x-userid',
    'cf_clearance',
    '__cf_bm',
)


class CookieRecord(SubsetModel):
    """One cookie entry from selenium-browser's profile state file.

    Subset of the file's full shape; we only need name/value plus expires
    for refresh detection. Domain/path/secure/etc. are present in the file
    but unused here (the SDK doesn't see them — only the formatted header).
    """

    name: str
    value: str
    expires: float = -1.0


class GrokCookies(ClosedModel):
    """Loaded cookie set for grok.com."""

    cookies: Sequence[CookieRecord]

    def by_name(self) -> Mapping[str, str]:
        return {c.name: c.value for c in self.cookies}

    def expires_by_name(self) -> Mapping[str, float]:
        return {c.name: c.expires for c in self.cookies}


def load_cookies(path: Path = DEFAULT_COOKIE_PATH) -> GrokCookies:
    """Read selenium-browser's profile state JSON and return cookies.

    Raises FileNotFoundError if the path doesn't exist (caller should
    invoke the bootstrap flow). Validation errors propagate.
    """
    data = json.loads(path.read_text())
    return GrokCookies(cookies=[CookieRecord(**c) for c in data.get('cookies', [])])


def format_cookie_header(cookies: GrokCookies | Mapping[str, str]) -> str:
    """Format cookies as a single ``Cookie:`` header value.

    Accepts either a parsed ``GrokCookies`` or a plain ``{name: value}``
    mapping. Output: ``name1=value1; name2=value2; ...`` matching the
    standard HTTP Cookie header format.
    """
    items = cookies.by_name().items() if isinstance(cookies, GrokCookies) else cookies.items()
    return '; '.join(f'{k}={v}' for k, v in items)


def missing_load_bearing(cookies: GrokCookies) -> Sequence[str]:
    """Names of load-bearing cookies absent from this set."""
    present = set(cookies.by_name().keys())
    return tuple(name for name in LOAD_BEARING_COOKIES if name not in present)


def expired_load_bearing(cookies: GrokCookies, *, now: datetime | None = None) -> Sequence[str]:
    """Names of load-bearing cookies that have expired.

    Cookies with ``expires == -1`` are session cookies (no expiration set);
    treated as never expired. Real expirations are unix timestamps.
    """
    cutoff = (now or datetime.now(UTC)).timestamp()
    return tuple(
        name for name, exp in cookies.expires_by_name().items() if name in LOAD_BEARING_COOKIES and 0 < exp < cutoff
    )
