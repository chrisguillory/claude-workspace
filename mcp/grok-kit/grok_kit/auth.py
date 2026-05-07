from __future__ import annotations

__all__ = [
    'DEFAULT_COOKIE_PATH',
    'LOAD_BEARING_COOKIES',
    'ImportResult',
    'expired_load_bearing',
    'format_cookie_header',
    'import_state',
    'load_state',
    'missing_load_bearing',
]

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from cc_lib.schemas.base import ClosedModel
from cc_lib.schemas.profile_state import ProfileState
from cc_lib.utils import get_claude_workspace_config_home_dir
from cc_lib.utils.atomic_write import atomic_write

DEFAULT_COOKIE_PATH = get_claude_workspace_config_home_dir() / 'mcp' / 'grok-kit' / 'cookies.json'

# Cookies required for grok.com's /rest/ endpoints, verified empirically by
# removing each in turn and observing 401:
#   sso, sso-rw  X SSO session tokens (~1 year)
#   x-userid     User UUID
#   cf_clearance Cloudflare bot-management challenge (12-72h, IP-bound)
#   __cf_bm      Cloudflare bot-management session (~30 min)
LOAD_BEARING_COOKIES: Sequence[str] = (
    'sso',
    'sso-rw',
    'x-userid',
    'cf_clearance',
    '__cf_bm',
)


class ImportResult(ClosedModel):
    """Outcome of ``import_state``: where it landed and what was found."""

    cookie_path: Path
    cookie_count: int
    missing_load_bearing: Sequence[str]
    expired_load_bearing: Sequence[str]


def load_state(path: Path = DEFAULT_COOKIE_PATH) -> ProfileState:
    """Read and validate a profile-state JSON file."""
    return ProfileState.model_validate_json(path.read_text())


def format_cookie_header(state: ProfileState) -> str:
    """Format a profile state's cookies as a single ``Cookie:`` header value."""
    return '; '.join(f'{c.name}={c.value}' for c in state.cookies)


def missing_load_bearing(state: ProfileState) -> Sequence[str]:
    """Names of load-bearing cookies absent from ``state``."""
    present = {c.name for c in state.cookies}
    return tuple(name for name in LOAD_BEARING_COOKIES if name not in present)


def expired_load_bearing(state: ProfileState, *, now: datetime | None = None) -> Sequence[str]:
    """Names of load-bearing cookies whose ``expires`` is in the past.

    Cookies with ``expires == -1`` are session cookies (no expiration set);
    treated as never expired. Real expirations are unix timestamps.
    """
    cutoff = (now or datetime.now(UTC)).timestamp()
    return tuple(c.name for c in state.cookies if c.name in LOAD_BEARING_COOKIES and 0 < c.expires < cutoff)


def import_state(state_path: Path, *, target_path: Path = DEFAULT_COOKIE_PATH) -> ImportResult:
    """Read a profile-state JSON and atomically install it at the canonical path."""
    state = load_state(state_path)
    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write(target_path, state.model_dump_json().encode(), mode=0o600)
    return ImportResult(
        cookie_path=target_path,
        cookie_count=len(state.cookies),
        missing_load_bearing=missing_load_bearing(state),
        expired_load_bearing=expired_load_bearing(state),
    )
