#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic>=2.0"]
# ///

"""Claude Code status line script.

Receives JSON on stdin from Claude Code after each assistant message.
Outputs ANSI-colored status information to display below the input area.

Strict Pydantic models validate the JSON schema — if Claude Code changes
the shape, this script fails immediately rather than showing stale data.

Install:
    chmod +x scripts/statusline.py
    # Add to ~/.claude/settings.json:
    # { "statusLine": { "type": "command", "command": "/path/to/statusline.py" } }
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import traceback
import types
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pydantic

# =============================================================================
# Pydantic Models — strict, fail-fast on schema drift
# =============================================================================


class StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


class ModelInfo(StrictModel):
    id: str
    display_name: str


class WorkspaceInfo(StrictModel):
    current_dir: str
    project_dir: str


class CostInfo(StrictModel):
    total_cost_usd: float | None = None
    total_duration_ms: float | None = None
    total_api_duration_ms: float | None = None
    total_lines_added: int | None = None
    total_lines_removed: int | None = None


class CurrentUsage(StrictModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class ContextWindow(StrictModel):
    total_input_tokens: int
    total_output_tokens: int
    context_window_size: int
    used_percentage: float | None = None
    remaining_percentage: float | None = None
    current_usage: CurrentUsage | None = None


class StatusLineInput(pydantic.BaseModel):
    """Top-level JSON received on stdin.

    Uses extra='allow' at top level since Claude Code may add new fields.
    Sub-objects use extra='forbid' to catch structural changes.
    """

    model_config = pydantic.ConfigDict(
        extra='allow',
        strict=True,
        frozen=True,
    )

    model: ModelInfo
    cwd: str
    workspace: WorkspaceInfo
    cost: CostInfo
    context_window: ContextWindow
    session_id: str
    transcript_path: str
    version: str
    exceeds_200k_tokens: bool
    # Conditional fields
    vim: Mapping[str, str] | None = None
    agent: Mapping[str, str] | None = None
    output_style: Mapping[str, str] | None = None


# =============================================================================
# ANSI Colors
# =============================================================================

CYAN = '\033[36m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RED = '\033[31m'
DIM = '\033[2m'
BOLD = '\033[1m'
RESET = '\033[0m'


def _osc8_link(url: str, text: str) -> str:
    """Wrap text in an OSC 8 clickable hyperlink (Cmd+click)."""
    return f'\033]8;;{url}\a{text}\033]8;;\a'


# =============================================================================
# Static Data Cache
#
# Email and subscription type don't change within a session.
# Cache in /tmp keyed by PID of the parent claude process to avoid
# stale data across account switches.
# =============================================================================

CACHE_PATH = Path('/tmp/claude-statusline-cache.json')
CACHE_TTL_SECONDS = 300  # 5 minutes


def _read_cached_static() -> Mapping[str, str] | None:
    """Read cached static data if fresh enough."""
    if not CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(CACHE_PATH.read_text())
        if not isinstance(raw, dict):
            return None
        ts = float(raw.get('_timestamp', 0))
        if (datetime.now(UTC).timestamp() - ts) < CACHE_TTL_SECONDS:
            return {k: str(v) for k, v in raw.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _write_cache(data: Mapping[str, str]) -> None:
    """Write static data to cache."""
    cache = {**data, '_timestamp': datetime.now(UTC).timestamp()}
    with contextlib.suppress(OSError):
        CACHE_PATH.write_text(json.dumps(cache))


def _read_static_data() -> Mapping[str, str]:
    """Read email and subscription from ~/.claude.json and keychain."""
    cached = _read_cached_static()
    if cached is not None:
        return cached

    data: dict[str, str] = {}

    # Email from ~/.claude.json
    config_path = Path.home() / '.claude.json'
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text())
            oa = config.get('oauthAccount', {})
            email = oa.get('emailAddress', '')
            if email:
                data['email'] = email
            org = oa.get('organizationName', '')
            if org:
                data['org'] = org
        except (json.JSONDecodeError, OSError):
            pass

    # Subscription type from keychain
    try:
        result = subprocess.run(
            [
                'security',
                'find-generic-password',
                '-a',
                os.environ.get('USER', ''),
                '-w',
                '-s',
                'Claude Code-credentials',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            kc = json.loads(result.stdout.strip())
            oauth = kc.get('claudeAiOauth', {})
            sub = oauth.get('subscriptionType', '')
            if sub:
                display = {
                    'free': 'Free',
                    'pro': 'Pro',
                    'team': 'Team',
                    'max': 'Max',
                    'enterprise': 'Enterprise',
                }.get(sub, sub)
                data['subscription'] = display
            tier = oauth.get('rateLimitTier', '')
            if tier:
                data['tier'] = tier
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass

    _write_cache(data)
    return data


def _git_branch() -> str:
    """Get current git branch (fast, no cache needed — git is quick)."""
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return ''


def _git_remote_url() -> str:
    """Get GitHub HTTPS URL for origin remote, or empty string."""
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Convert git@github.com:user/repo.git → https://github.com/user/repo
            if url.startswith('git@github.com:'):
                url = 'https://github.com/' + url[15:]
            if url.endswith('.git'):
                url = url[:-4]
            if 'github.com' in url:
                return url
    except (subprocess.TimeoutExpired, OSError):
        pass
    return ''


# =============================================================================
# Formatting
# =============================================================================


def _context_bar(pct: float) -> str:
    """Render a 10-char progress bar with color."""
    if pct >= 90:
        color = RED
    elif pct >= 70:
        color = YELLOW
    else:
        color = GREEN

    filled = int(pct / 10)
    bar = '█' * filled + '░' * (10 - filled)
    return f'{color}{bar}{RESET} {pct:.0f}%'


def _format_duration(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    total_seconds = int(ms / 1000)
    if total_seconds < 60:
        return f'{total_seconds}s'
    mins = total_seconds // 60
    secs = total_seconds % 60
    if mins < 60:
        return f'{mins}m{secs:02d}s'
    hours = mins // 60
    remaining_mins = mins % 60
    return f'{hours}h{remaining_mins:02d}m'


def _format_cost(usd: float) -> str:
    """Format cost with color based on amount."""
    if usd >= 5.0:
        color = RED
    elif usd >= 1.0:
        color = YELLOW
    else:
        color = DIM
    return f'{color}${usd:.2f}{RESET}'


def _format_tokens(n: int) -> str:
    """Format token count as compact string."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.0f}k'
    return str(n)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    log_path = Path('/tmp/claude-statusline-error.log')

    try:
        raw = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        msg = f'{datetime.now(UTC).isoformat()}\nJSON decode failed: {e}\n\n'
        with contextlib.suppress(OSError):
            log_path.write_text(msg)
        print(f'{RED}StatusLine: invalid JSON{RESET}', file=sys.stderr)
        return

    try:
        data = StatusLineInput.model_validate(raw)
    except pydantic.ValidationError as e:
        # Log raw JSON and errors for debugging
        error_msg = f'{datetime.now(UTC).isoformat()}\nValidation failed:\n'
        for err in e.errors():
            loc = '.'.join(str(x) for x in err['loc'])
            error_msg += f'  {loc}: {err["msg"]}\n'
        error_msg += '\nRaw JSON:\n'
        error_msg += json.dumps(raw, indent=2)
        error_msg += '\n\n'

        with contextlib.suppress(OSError):
            log_path.write_text(error_msg)

        print(f'{RED}StatusLine: validation failed — see {log_path}{RESET}', file=sys.stderr)
        return

    static = _read_static_data()
    branch = _git_branch()
    remote_url = _git_remote_url()

    # ── Line 1: Identity + Model ─────────────────────────────────────────
    parts: list[str] = []

    # Model
    parts.append(f'{CYAN}[{data.model.display_name}]{RESET}')

    # Session ID — space-separated so UUID is double-clickable, links to transcript
    transcript_url = f'file://{data.transcript_path}'
    parts.append(f'{DIM}session_id:{RESET} {_osc8_link(transcript_url, data.session_id)}')

    # Account
    email = static.get('email', '')
    sub = static.get('subscription', '')
    if email:
        parts.append(f'{DIM}{email}{RESET}')
    if sub:
        org = static.get('org', '')
        if org:
            parts.append(f'{DIM}{sub}·{org}{RESET}')
        else:
            parts.append(f'{DIM}{sub}{RESET}')

    # Git branch — links to GitHub branch page
    if branch:
        if remote_url:
            branch_link = _osc8_link(f'{remote_url}/tree/{branch}', branch)
            parts.append(f'{DIM}⎇{RESET} {branch_link}')
        else:
            parts.append(f'{DIM}⎇ {branch}{RESET}')

    # Vim mode
    if data.vim and 'mode' in data.vim:
        mode = data.vim['mode']
        mode_color = GREEN if mode == 'NORMAL' else YELLOW
        parts.append(f'{mode_color}{mode}{RESET}')

    print(' '.join(parts))

    # ── Line 2: Metrics ──────────────────────────────────────────────────
    metrics: list[str] = []

    # Context bar
    pct = data.context_window.used_percentage or 0
    metrics.append(_context_bar(pct))

    # Token counts
    ctx = data.context_window
    if ctx.current_usage is not None:
        in_tokens = _format_tokens(ctx.current_usage.input_tokens)
        out_tokens = _format_tokens(ctx.current_usage.output_tokens)
        metrics.append(f'{DIM}in:{in_tokens} out:{out_tokens}{RESET}')

    # Cost
    cost = data.cost.total_cost_usd
    if cost is not None:
        metrics.append(_format_cost(cost))

    # Duration
    duration = data.cost.total_duration_ms
    if duration is not None and duration > 0:
        metrics.append(f'{DIM}{_format_duration(duration)}{RESET}')

    # Lines changed
    added = data.cost.total_lines_added or 0
    removed = data.cost.total_lines_removed or 0
    if added or removed:
        metrics.append(f'{GREEN}+{added}{RESET}{RED}-{removed}{RESET}')

    print(' │ '.join(metrics))

    # ── Line 3: Workspace + Version ─────────────────────────────────────
    line3: list[str] = []

    line3.append(f'{DIM}v{data.version}{RESET}')

    # Show cwd; if project_dir differs, show both
    if data.workspace.project_dir != data.workspace.current_dir:
        line3.append(f'{DIM}cwd:{RESET} {data.cwd}')
        line3.append(f'{DIM}project:{RESET} {data.workspace.project_dir}')
    else:
        line3.append(f'{DIM}cwd:{RESET} {data.cwd}')

    # Rate limit tier
    tier = static.get('tier', '')
    if tier:
        line3.append(f'{DIM}tier:{RESET} {tier}')

    # Transcript path
    line3.append(f'{DIM}transcript:{RESET} {_osc8_link(transcript_url, data.transcript_path)}')

    print(' │ '.join(line3))

    # ── Line 4: Context Window Detail ───────────────────────────────────
    line4: list[str] = []

    # Cumulative totals
    ctx_in = _format_tokens(ctx.total_input_tokens)
    ctx_out = _format_tokens(ctx.total_output_tokens)
    line4.append(f'{DIM}total_in:{RESET} {ctx_in}')
    line4.append(f'{DIM}total_out:{RESET} {ctx_out}')

    # Context window size
    ctx_size = _format_tokens(ctx.context_window_size)
    line4.append(f'{DIM}window:{RESET} {ctx_size}')

    # Remaining percentage
    remaining = ctx.remaining_percentage
    if remaining is not None:
        rem_color = GREEN if remaining > 30 else YELLOW if remaining > 10 else RED
        line4.append(f'{rem_color}{remaining:.0f}% remaining{RESET}')

    # Exceeds 200k warning
    if data.exceeds_200k_tokens:
        line4.append(f'{RED}⚠ >200k tokens{RESET}')

    # Cache tokens (from current usage)
    if ctx.current_usage is not None:
        cache_create = ctx.current_usage.cache_creation_input_tokens
        cache_read = ctx.current_usage.cache_read_input_tokens
        if cache_create or cache_read:
            line4.append(f'{DIM}cache: +{_format_tokens(cache_create)} ↺{_format_tokens(cache_read)}{RESET}')

    # API duration (distinct from wall clock duration)
    api_dur = data.cost.total_api_duration_ms
    if api_dur is not None and api_dur > 0:
        line4.append(f'{DIM}api:{_format_duration(api_dur)}{RESET}')

    print(' │ '.join(line4))


def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None) -> None:
    if issubclass(exc_type, Exception):
        print(f'{RED}StatusLine crashed: {exc_type.__name__}: {exc_value}{RESET}', file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    else:
        sys.__excepthook__(exc_type, exc_value, exc_tb)


if __name__ == '__main__':
    sys.excepthook = _excepthook
    main()
