#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "psutil>=5.9",
#     "pydantic>=2.0",
# ]
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
import time
import traceback
import types
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psutil
import pydantic
import pydantic.alias_generators

# =============================================================================
# Pydantic Models — strict, fail-fast on schema drift
# =============================================================================


class StrictModel(pydantic.BaseModel):
    """Shared base: no extra fields, strict type coercion, immutable."""

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


class _ExternalModel(pydantic.BaseModel):
    """Base for external data we don't control. Ignores unknown fields."""

    model_config = pydantic.ConfigDict(extra='ignore', frozen=True)


# =============================================================================
# Credential Models — External Data (login files, config, keychain)
#
# Lightweight projections of external schemas. Only includes fields the
# statusline needs. Uses extra='ignore' since these files have many fields
# we don't care about (tokens, scopes, display_name, etc.).
# =============================================================================


class LoginFileOAuthAccount(_ExternalModel):
    """Login file's oauth_account section (snake_case from disk)."""

    email_address: str = ''
    organization_uuid: str = ''
    billing_type: str | None = None


class LoginFileClaudeAiOAuth(_ExternalModel):
    """Login file's claude_ai_oauth section (snake_case from disk)."""

    subscription_type: str | None = None
    rate_limit_tier: str | None = None


class LoginFile(_ExternalModel):
    """Saved login file from ~/.claude-workspace/logins/*.json."""

    name: str = ''
    oauth_account: LoginFileOAuthAccount | None = None
    claude_ai_oauth: LoginFileClaudeAiOAuth | None = None
    setup_token: str | None = None


class ConfigOAuthAccount(_ExternalModel):
    """~/.claude.json oauthAccount section (camelCase from disk)."""

    model_config = pydantic.ConfigDict(
        extra='ignore',
        frozen=True,
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    email_address: str = ''
    organization_uuid: str = ''
    billing_type: str = ''


class KeychainClaudeAiOAuth(_ExternalModel):
    """Keychain claudeAiOauth section (camelCase from disk)."""

    model_config = pydantic.ConfigDict(
        extra='ignore',
        frozen=True,
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    subscription_type: str = ''
    rate_limit_tier: str = ''
    access_token: str = ''


class KeychainData(_ExternalModel):
    """Top-level keychain JSON from macOS security command."""

    model_config = pydantic.ConfigDict(
        extra='ignore',
        frozen=True,
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    claude_ai_oauth: KeychainClaudeAiOAuth | None = None


class SwitchPendingMarker(_ExternalModel):
    """The .switch-pending file written by claude-login (camelCase from disk)."""

    model_config = pydantic.ConfigDict(
        extra='ignore',
        frozen=True,
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    email_address: str = ''
    billing_type: str = ''


# =============================================================================
# Credential Models — Internal (resolved state flowing through pipeline)
# =============================================================================


class ResolvedCredentials(pydantic.BaseModel):
    """Resolved credential state flowing through the pipeline.

    Replaces Mapping[str, str] with typed fields. Uses camelCase JSON aliases
    for backward compatibility with existing cache and snapshot files on disk.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
        alias_generator=pydantic.alias_generators.to_camel,
        populate_by_name=True,
    )

    email_address: str = ''
    organization_uuid: str = ''
    billing_type: str = ''
    subscription: str = ''
    tier: str = ''
    login_name: str = ''
    auth_method: str = ''


class CachedCredentials(StrictModel):
    """Cache wrapper: ResolvedCredentials + timestamp for TTL and invalidation."""

    timestamp: float
    credentials: ResolvedCredentials


class CredentialSnapshot(pydantic.BaseModel):
    """Per-session credential snapshot, persisted to disk.

    Not strict — existing snapshot files may have numeric claude_pid that
    needs coercion. credentials sub-object handles its own alias mapping.
    """

    model_config = pydantic.ConfigDict(extra='forbid', frozen=True)

    session_id: str
    claude_pid: int
    created_at: str
    credentials: ResolvedCredentials


def _extract_plan_info(login: LoginFile) -> tuple[str, str, str]:
    """Extract (login_name, subscription_display, tier) from a LoginFile."""
    login_name = login.name
    subscription = ''
    tier = ''
    if login.claude_ai_oauth is not None:
        sub_raw = login.claude_ai_oauth.subscription_type or ''
        if sub_raw:
            subscription = SUBSCRIPTION_DISPLAY.get(sub_raw, sub_raw)
        tier = login.claude_ai_oauth.rate_limit_tier or ''
    return login_name, subscription, tier


# =============================================================================
# Health Monitoring Models
#
# Tracks process health across statusline invocations via a per-session sidecar
# file. Each invocation is a fresh Python process, so all cross-invocation state
# (CPU deltas, memory trend, peak tracking) must be persisted to disk.
#
# HealthSample: single point-in-time measurement from psutil
# HealthSidecar: ring buffer of recent samples + session-level aggregates
# =============================================================================


class HealthSample(StrictModel):
    """Point-in-time process health measurement from psutil."""

    ts: float  # time.time() — comparable across processes (unlike monotonic)
    rss_bytes: int  # memory_info().rss
    cpu_user: float  # cpu_times().user (cumulative seconds)
    cpu_system: float  # cpu_times().system (cumulative seconds)
    num_fds: int  # num_fds()
    num_children: int  # len(children(recursive=True))
    num_zombies: int  # children with STATUS_ZOMBIE or STATUS_DEAD
    tree_rss_bytes: int  # sum of RSS across all child processes


class HealthSidecar(StrictModel):
    """Per-session health tracking, persisted between statusline invocations.

    Ring buffer of recent samples enables delta CPU% calculation and memory
    trend detection. Reset when claude_pid changes (session resumed with new
    process) to avoid stale deltas.
    """

    session_id: str
    claude_pid: int  # Reset sidecar when PID changes
    peak_rss_bytes: int  # High water mark across session lifetime
    samples: Sequence[HealthSample]  # Capped at MAX_HEALTH_SAMPLES


# =============================================================================
# Status Line Input Models — strict, fail-fast on schema drift
# =============================================================================


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


# =============================================================================
# Health Monitoring Thresholds
#
# Calibrated against observed Claude Code behavior on macOS (Apple Silicon).
# Claude Code is Node.js — single-threaded event loop, so CPU% is percentage
# of ONE core (0-100%). The documented problems and their thresholds:
#
# CPU:  Ink/React TUI busy-wait pegs one core at ~100% when idle.
#       Normal idle is 0-5%. (anthropics/claude-code#22275, #22131)
# Mem:  Node.js heap leak grows from ~300MB to multi-GB over hours.
#       Healthy session is 300-600MB RSS. (anthropics/claude-code#5771)
# FDs:  File descriptor handles to .claude config files never closed.
#       Healthy session has ~45 FDs. (anthropics/claude-code#21701, #23645)
# Kids: MCP servers and subagents accumulate if not cleaned up.
#       Healthy session has 3-8 children. (anthropics/claude-code#25180)
# =============================================================================

# Memory RSS thresholds (bytes)
MEMORY_WARN_BYTES = 1_000_000_000  # 1GB — above typical healthy range
MEMORY_CRIT_BYTES = 3_000_000_000  # 3GB — likely memory leak in progress

# CPU percentage of one core (0-100%)
CPU_WARN_PERCENT = 30.0  # Sustained background work, worth watching
CPU_CRIT_PERCENT = 80.0  # Stuck busy-wait loop or runaway process

# File descriptors — anomaly display trigger
FD_WARN_COUNT = 100  # Well above normal (~45 observed healthy)

# Child processes (recursive) — anomaly display triggers
CHILDREN_WARN_COUNT = 30  # Above observed baseline of ~20 (MCP servers + subprocesses)
CHILDREN_CRIT_COUNT = 50  # Process accumulation bug

# Process tree memory — aggregate RSS across all children. Catches MCP server
# memory leaks where the main Claude process looks fine but children balloon.
TREE_RSS_WARN_BYTES = 2_000_000_000  # 2GB — tree consuming significant system memory

# Growth rate alerts — catch leaks BEFORE they hit absolute thresholds.
# Rates are per hour, computed from oldest vs newest persisted sample.
# Only shown after MIN_TREND_SAMPLES spanning MIN_TREND_SPAN_SECONDS.
FD_GROWTH_RATE_WARN = 20  # FDs/hour — steady leak pattern
CHILDREN_GROWTH_RATE_WARN = 10  # children/hour — process accumulation pattern

# Sampling — the statusline fires multiple times per assistant turn (streaming
# start, tool call boundaries, completion), often 0.3s apart. Only persist to
# the ring buffer at meaningful intervals so CPU% deltas are always computable
# and the buffer spans real time, not sub-second bursts. Live psutil data is
# always collected for display regardless of this interval.
MIN_SAMPLE_INTERVAL = 2.0  # Minimum seconds between persisted samples

# Trend detection — minimum data before showing memory trend
MAX_HEALTH_SAMPLES = 60  # Ring buffer size (~2 hours at 1 sample per 2s minimum)
MIN_TREND_SAMPLES = 5  # Minimum samples before showing trend arrow
MIN_TREND_SPAN_SECONDS = 120.0  # Minimum time span for meaningful trend
MEM_TREND_GROWTH_PCT = 20.0  # % growth between oldest/newest to show ↑
MEM_TREND_SHRINK_PCT = 20.0  # % shrink between oldest/newest to show ↓


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

CONFIG_PATH = Path.home() / '.claude.json'
SCRIPT_PATH = Path(__file__).resolve()
CACHE_PATH = Path.home() / '.claude-workspace' / 'statusline-cache.json'
CACHE_TTL_SECONDS = 300  # 5 minutes
SNAPSHOT_DIR = Path.home() / '.claude-workspace' / 'statusline-snapshots'
SWITCH_PENDING_PATH = Path.home() / '.claude-workspace' / '.switch-pending'
LOGINS_DIR = Path.home() / '.claude-workspace' / 'logins'

SUBSCRIPTION_DISPLAY: Mapping[str, str] = {
    'free': 'Free',
    'pro': 'Pro',
    'team': 'Team',
    'max': 'Max',
    'enterprise': 'Enterprise',
}


def _max_mtime() -> float:
    """Return the most recent mtime of config, script, or login files."""
    mtime = 0.0
    for path in (CONFIG_PATH, SCRIPT_PATH):
        with contextlib.suppress(OSError):
            mtime = max(mtime, path.stat().st_mtime)
    if LOGINS_DIR.is_dir():
        for path in LOGINS_DIR.glob('*.json'):
            with contextlib.suppress(OSError):
                mtime = max(mtime, path.stat().st_mtime)
    return mtime


def _read_cached_static() -> CachedCredentials | None:
    """Read cached credentials if fresh and neither config nor script changed."""
    if not CACHE_PATH.exists():
        return None
    try:
        cached = CachedCredentials.model_validate_json(CACHE_PATH.read_text())
        if (datetime.now(UTC).timestamp() - cached.timestamp) > CACHE_TTL_SECONDS:
            return None
        if _max_mtime() > cached.timestamp:
            return None
        return cached
    except (pydantic.ValidationError, json.JSONDecodeError, OSError):
        return None


def _read_cache_timestamp() -> float:
    """Read cache timestamp without TTL/invalidation checks (for debug display)."""
    try:
        cached = CachedCredentials.model_validate_json(CACHE_PATH.read_text())
        return cached.timestamp
    except (pydantic.ValidationError, json.JSONDecodeError, OSError):
        return 0.0


def _write_cache(creds: ResolvedCredentials) -> None:
    """Write credentials to cache with current timestamp.

    Uses default snake_case serialization (no by_alias). Snapshot files use
    by_alias=True for camelCase backward compatibility. Both parse correctly
    due to populate_by_name=True on ResolvedCredentials.
    """
    cached = CachedCredentials(
        timestamp=datetime.now(UTC).timestamp(),
        credentials=creds,
    )
    with contextlib.suppress(OSError):
        CACHE_PATH.write_text(cached.model_dump_json())
        CACHE_PATH.chmod(0o600)


def _find_claude_pid() -> int:
    """Find Claude Code PID by walking up the process tree via psutil.

    Uses psutil instead of spawning `ps` subprocesses — ~0.4ms vs ~6ms per
    ancestor hop. Checks exe() path because psutil.name() returns the version
    number (e.g., '2.1.44'), not 'claude', on macOS.
    """
    current = os.getppid()
    for _ in range(20):
        try:
            proc = psutil.Process(current)
            exe = proc.exe()
            if 'claude' in exe.lower():
                return current
            ppid = proc.ppid()
            if ppid == 0:
                break
            current = ppid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    raise RuntimeError('Could not find Claude Code process in parent tree')


def _snapshot_path(session_id: str) -> Path:
    return SNAPSHOT_DIR / f'{session_id}.json'


def _atomic_write_json(path: Path, data: str) -> None:
    """Atomically write JSON string to file (temp → fsync → rename, 0o600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    try:
        with tmp.open('w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        path.chmod(0o600)
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _is_switch_pending(disk_data: ResolvedCredentials) -> bool:
    """Check if switch-login's marker matches current disk state.

    Compares identity fields only — must match the same keys used for
    account_changed detection in _get_active_credentials.
    """
    try:
        marker = SwitchPendingMarker.model_validate_json(SWITCH_PENDING_PATH.read_text())
    except OSError:
        return False
    return marker.email_address == disk_data.email_address and marker.billing_type == disk_data.billing_type


def _get_active_credentials(session_id: str, claude_pid: int) -> tuple[ResolvedCredentials, bool]:
    """Return (credentials_to_display, switch_pending).

    On first call for a session, snapshots the current disk state.
    On subsequent calls, compares snapshot vs disk to detect mid-session switches.
    """
    snap_file = _snapshot_path(session_id)
    disk_data = _read_static_data()

    snap: CredentialSnapshot | None = None
    if snap_file.exists():
        try:
            snap = CredentialSnapshot.model_validate_json(snap_file.read_text())
        except OSError:
            snap_file.unlink(missing_ok=True)

    if snap is not None:
        # PID changed = session was resumed with new process → re-snapshot
        if snap.claude_pid != claude_pid:
            snap_file.unlink(missing_ok=True)
            SWITCH_PENDING_PATH.unlink(missing_ok=True)
        else:
            snap_creds = snap.credentials
            # Only compare identity fields — plan info can change without
            # account change (e.g., reverse mapping enriches setup-token data).
            account_changed = (
                snap_creds.email_address != disk_data.email_address or snap_creds.billing_type != disk_data.billing_type
            )
            if account_changed:
                if _is_switch_pending(disk_data):
                    return snap_creds, True
                # /login or other cause — update snapshot to match new reality
                new_snap = CredentialSnapshot(
                    session_id=session_id,
                    claude_pid=claude_pid,
                    created_at=datetime.now(UTC).isoformat(),
                    credentials=disk_data,
                )
                _atomic_write_json(snap_file, new_snap.model_dump_json(by_alias=True, indent=2))
                return disk_data, False
            # Same account — update snapshot with latest display data
            # (reverse mapping may have enriched subscription/tier)
            if snap_creds != disk_data:
                new_snap = CredentialSnapshot(
                    session_id=session_id,
                    claude_pid=claude_pid,
                    created_at=snap.created_at,
                    credentials=disk_data,
                )
                _atomic_write_json(snap_file, new_snap.model_dump_json(by_alias=True, indent=2))
            return disk_data, False

    # First invocation (or PID changed) — create snapshot
    new_snap = CredentialSnapshot(
        session_id=session_id,
        claude_pid=claude_pid,
        created_at=datetime.now(UTC).isoformat(),
        credentials=disk_data,
    )
    _atomic_write_json(snap_file, new_snap.model_dump_json(by_alias=True, indent=2))

    # Clean up orphans on first invocation
    _cleanup_orphan_snapshots(session_id)

    return disk_data, False


def _cleanup_orphan_snapshots(current_session_id: str) -> None:
    """Remove snapshot and health sidecar files whose Claude PID is no longer running."""
    if not SNAPSHOT_DIR.exists():
        return
    for path in SNAPSHOT_DIR.glob('*.json'):
        if path.stem.startswith(current_session_id):
            continue
        try:
            # Both CredentialSnapshot and HealthSidecar have claude_pid
            if path.stem.endswith('-health'):
                pid = HealthSidecar.model_validate_json(path.read_text()).claude_pid
            else:
                pid = CredentialSnapshot.model_validate_json(path.read_text()).claude_pid
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Process exists but owned by another user


def _iter_logins() -> Sequence[LoginFile]:
    """Read all saved login files from disk."""
    if not LOGINS_DIR.is_dir():
        return []
    results: list[LoginFile] = []
    for path in LOGINS_DIR.glob('*.json'):
        try:
            results.append(LoginFile.model_validate_json(path.read_text()))
        except OSError:
            continue
    return results


def _reverse_map_setup_token(access_token: str) -> ResolvedCredentials | None:
    """Match a keychain accessToken against saved logins' setup_token fields."""
    if not access_token:
        return None
    for login in _iter_logins():
        if login.setup_token == access_token:
            login_name, subscription, tier = _extract_plan_info(login)
            return ResolvedCredentials(login_name=login_name, subscription=subscription, tier=tier)
    return None


def _resolve_login(email: str, billing_type: str, org_uuid: str) -> ResolvedCredentials | None:
    """Match current credentials to a saved login by organization UUID.

    Authoritative match — overrides setup-token reverse mapping when both run.
    """
    if not email:
        return None
    for login in _iter_logins():
        oa = login.oauth_account
        if oa is not None and (
            oa.email_address == email and oa.billing_type == billing_type and oa.organization_uuid == org_uuid
        ):
            login_name, subscription, tier = _extract_plan_info(login)
            return ResolvedCredentials(login_name=login_name, subscription=subscription, tier=tier)
    return None


def _read_static_data() -> ResolvedCredentials:
    """Read email and subscription from ~/.claude.json and keychain."""
    cached = _read_cached_static()
    if cached is not None:
        return cached.credentials

    email_address = ''
    organization_uuid = ''
    billing_type = ''
    subscription = ''
    tier = ''
    login_name = ''
    auth_method = ''

    # Identity from ~/.claude.json (external data — defensive parsing)
    if CONFIG_PATH.is_file():
        try:
            config = json.loads(CONFIG_PATH.read_text())
            oa_raw = config.get('oauthAccount')
            if isinstance(oa_raw, dict):
                oa = ConfigOAuthAccount.model_validate(oa_raw)
                email_address = oa.email_address
                organization_uuid = oa.organization_uuid
                billing_type = oa.billing_type
        except (pydantic.ValidationError, json.JSONDecodeError, OSError):
            pass

    # Subscription from keychain (external data — defensive parsing)
    try:
        result = subprocess.run(
            ['security', 'find-generic-password', '-w', '-s', 'Claude Code-credentials'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            kc = KeychainData.model_validate_json(result.stdout.strip())
            oauth = kc.claude_ai_oauth
            if oauth is not None:
                if oauth.subscription_type:
                    subscription = SUBSCRIPTION_DISPLAY.get(oauth.subscription_type, oauth.subscription_type)
                if oauth.rate_limit_tier:
                    tier = oauth.rate_limit_tier

                # Setup-token reverse mapping: when subscriptionType is null
                # (setup-token in keychain), look up plan info from saved logins.
                if not oauth.subscription_type and oauth.access_token:
                    login_data = _reverse_map_setup_token(oauth.access_token)
                    if login_data is not None:
                        login_name = login_data.login_name or login_name
                        subscription = login_data.subscription or subscription
                        tier = login_data.tier or tier
                        if not auth_method:
                            auth_method = 'setup-token'
    except (subprocess.TimeoutExpired, pydantic.ValidationError, json.JSONDecodeError, OSError):
        pass

    # Resolve login by org UUID — authoritative match that overrides
    # setup-token reverse mapping (which can match the wrong account)
    login_info = _resolve_login(email_address, billing_type, organization_uuid)
    if login_info is not None:
        login_name = login_info.login_name or login_name
        subscription = login_info.subscription or subscription
        tier = login_info.tier or tier

    creds = ResolvedCredentials(
        email_address=email_address,
        organization_uuid=organization_uuid,
        billing_type=billing_type,
        subscription=subscription,
        tier=tier,
        login_name=login_name,
        auth_method=auth_method,
    )
    _write_cache(creds)
    return creds


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


def _format_plan_compact(sub: str, tier: str) -> str:
    """Extract tier multiplier (5x, 20x) or fall back to plan name (Pro).

    Used alongside login IDs where account type is already visible in the ID
    (e.g., --Personal, --Team), so only the multiplier adds information.
    """
    if tier.startswith('default_claude_'):
        raw = tier.removeprefix('default_claude_')
        parts = raw.split('_', 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]  # "5x", "20x"
    # No multiplier — show plan name only if it adds info beyond the login ID
    if sub and sub.lower() not in ('max', 'team'):
        return sub.title()  # "Pro", "Free"
    return ''


def _context_bar(pct: float, window_size: int) -> str:
    """Render a 10-char progress bar with color and estimated token usage."""
    if pct >= 90:
        color = RED
    elif pct >= 70:
        color = YELLOW
    else:
        color = GREEN

    filled = int(pct / 10)
    bar = '█' * filled + '░' * (10 - filled)
    used = _format_tokens(int(pct / 100 * window_size))
    total = _format_tokens(window_size)
    return f'{color}{bar}{RESET} {pct:.0f}% {DIM}({used}/{total}){RESET}'


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


def _shorten_path(path: str) -> str:
    """Replace home directory prefix with ~ for display."""
    home = str(Path.home())
    if path.startswith(home):
        return '~' + path[len(home) :]
    return path


def _format_tokens(n: int) -> str:
    """Format token count as compact string."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.0f}k'
    return str(n)


# =============================================================================
# Process Health Monitoring
#
# Collects CPU, memory, file descriptor, and child process metrics via psutil.
# Cross-invocation state (for CPU% deltas and memory trend) persisted in a
# per-session sidecar file alongside existing credential snapshots.
#
# Design rationale:
# - Each statusline invocation is a FRESH Python process — no in-memory state.
# - CPU% requires delta between two readings: we store cumulative cpu_times()
#   and wall timestamp, then compute (delta_cpu / delta_wall) * 100 next time.
# - Memory trend compares oldest vs newest sample in the ring buffer, but only
#   after MIN_TREND_SAMPLES spanning MIN_TREND_SPAN_SECONDS to avoid noise.
# - Child process scan uses recursive=True to catch the full tree (MCP servers,
#   subagents, and their children). This is ~15ms — acceptable for a statusline.
# - All psutil calls wrapped in try/except for NoSuchProcess and AccessDenied.
#   Per-child errors are handled individually — one dead child won't abort the
#   survey. If the Claude process itself dies, we return None gracefully.
# =============================================================================


def _health_sidecar_path(session_id: str) -> Path:
    """Sidecar file for process health, stored alongside credential snapshots."""
    return SNAPSHOT_DIR / f'{session_id}-health.json'


def _load_health_sidecar(session_id: str) -> HealthSidecar | None:
    """Load previous health sidecar, or None if missing/corrupt.

    Returns None on any failure (missing file, corrupt JSON, schema mismatch).
    Callers treat None as "first invocation" and bootstrap from scratch.
    """
    path = _health_sidecar_path(session_id)
    if not path.exists():
        return None
    try:
        return HealthSidecar.model_validate_json(path.read_text())
    except (pydantic.ValidationError, json.JSONDecodeError, OSError):
        return None


def _save_health_sidecar(sidecar: HealthSidecar) -> None:
    """Atomically save health sidecar via shared atomic write helper."""
    path = _health_sidecar_path(sidecar.session_id)
    _atomic_write_json(path, sidecar.model_dump_json())


def _collect_health_sample(claude_pid: int) -> HealthSample | None:
    """Collect current health metrics from the Claude process via psutil.

    Returns None if the process no longer exists or is inaccessible.
    Uses oneshot() context manager for efficient batched syscalls (~0.2ms).
    """
    try:
        proc = psutil.Process(claude_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    try:
        with proc.oneshot():
            cpu_t = proc.cpu_times()
            mem = proc.memory_info()
            fd_count = proc.num_fds()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

    # Child process survey (~15ms for typical tree of ~20 children).
    # Uses recursive=True to catch the full tree — users report 200+ processes
    # accumulating as MCP servers and subagents spawn their own children.
    # Collects per-child RSS (~0.05ms each) and zombie status in a single pass.
    child_count = 0
    zombie_count = 0
    tree_rss = 0
    try:
        children = proc.children(recursive=True)
        child_count = len(children)
        for child in children:
            try:
                # macOS may report zombies as STATUS_DEAD depending on psutil version
                if child.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    zombie_count += 1
                else:
                    tree_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Child died between enumeration and status/memory check
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass  # Parent died during child enumeration

    return HealthSample(
        ts=time.time(),
        rss_bytes=mem.rss,
        cpu_user=cpu_t.user,
        cpu_system=cpu_t.system,
        num_fds=fd_count,
        num_children=child_count,
        num_zombies=zombie_count,
        tree_rss_bytes=tree_rss,
    )


def _update_health_sidecar(
    session_id: str,
    claude_pid: int,
    sample: HealthSample,
) -> HealthSidecar:
    """Conditionally persist sample to sidecar, decoupling display from history.

    The statusline fires multiple times per assistant turn (streaming start,
    tool call boundaries, completion) — often 0.3s apart. If we persisted every
    invocation, the ring buffer would fill with sub-second noise: 60 samples
    spanning 30 seconds instead of ~2 hours of meaningful history.

    Solution: two separate concerns with different cadences.
      - Display: always uses the live sample (fresh mem/FDs/children).
      - History: only persisted at MIN_SAMPLE_INTERVAL gaps, so CPU% deltas
        between consecutive persisted samples are always meaningful, and the
        ring buffer spans real wall-clock time for trend analysis.

    First invocation and PID changes always persist immediately.
    """
    existing = _load_health_sidecar(session_id)

    # PID mismatch means session was resumed with a new process — start fresh
    if existing is not None and existing.claude_pid != claude_pid:
        existing = None

    if existing is None:
        # First invocation — always persist
        sidecar = HealthSidecar(
            session_id=session_id,
            claude_pid=claude_pid,
            peak_rss_bytes=sample.rss_bytes,
            samples=[sample],
        )
        _save_health_sidecar(sidecar)
        return sidecar

    # Check if enough time has passed to persist this sample
    last_ts = existing.samples[-1].ts if existing.samples else 0.0
    if sample.ts - last_ts < MIN_SAMPLE_INTERVAL:
        # Too soon — return existing sidecar with updated peak (in memory only,
        # not persisted) so display still reflects current peak awareness.
        return HealthSidecar(
            session_id=existing.session_id,
            claude_pid=existing.claude_pid,
            peak_rss_bytes=max(existing.peak_rss_bytes, sample.rss_bytes),
            samples=existing.samples,
        )

    # Enough time passed — persist this sample
    samples = [*existing.samples, sample]
    if len(samples) > MAX_HEALTH_SAMPLES:
        samples = samples[-MAX_HEALTH_SAMPLES:]
    sidecar = HealthSidecar(
        session_id=session_id,
        claude_pid=claude_pid,
        peak_rss_bytes=max(existing.peak_rss_bytes, sample.rss_bytes),
        samples=samples,
    )
    _save_health_sidecar(sidecar)
    return sidecar


def _compute_cpu_percent(current: HealthSample, previous: HealthSample) -> float | None:
    """Average CPU% of one core between two samples via delta cpu_times.

    Node.js is single-threaded — the meaningful signal is whether one core is
    pegged (>80%), not total system load. Returns None if samples are too close
    together (<0.5s) to produce a meaningful delta.
    """
    dt = current.ts - previous.ts
    if dt < 0.5:
        return None
    cpu_delta = (current.cpu_user - previous.cpu_user) + (current.cpu_system - previous.cpu_system)
    return max(0.0, min(cpu_delta / dt * 100.0, 100.0))


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable memory (e.g., 451M, 1.2G)."""
    if n >= 1_000_000_000:
        return f'{n / 1_000_000_000:.1f}G'
    return f'{n // 1_000_000}M'


def _color_for_value(value: float, warn: float, crit: float) -> str:
    """Return ANSI color based on threshold: green < warn, yellow < crit, red."""
    if value >= crit:
        return RED
    if value >= warn:
        return YELLOW
    return GREEN


def _format_health(sample: HealthSample, sidecar: HealthSidecar) -> str:
    """Format process health metrics for line 1 display.

    Uses two data sources with different roles:
      - sample: live psutil snapshot — always fresh for mem/FDs/children display.
      - sidecar.samples: persisted ring buffer at ≥2s intervals — used for CPU%
        deltas and memory trend, where meaningful time gaps are required.

    Tier 1 (always shown): mem and cpu, color-coded by severity.
    Tier 2 (anomaly-only): zombies, excess children, FD count — shown only
      when thresholds are exceeded, keeping the line clean during normal operation.
    Tier 3 (trend): peak memory and trend arrow — shown after enough samples
      to produce meaningful data (MIN_TREND_SAMPLES over MIN_TREND_SPAN_SECONDS).
    """
    parts: list[str] = []

    # ── Tier 1: Memory (always shown) ──
    mem_color = _color_for_value(sample.rss_bytes, MEMORY_WARN_BYTES, MEMORY_CRIT_BYTES)
    mem_str = f'{mem_color}mem: {_format_bytes(sample.rss_bytes)}{RESET}'

    # ── Tier 3: Memory trend arrow ──
    # Compare oldest vs newest sample to detect sustained growth or shrinkage.
    # Requires enough samples spanning enough time to avoid noise from transient
    # spikes (e.g., a large context window load that gets GC'd).
    samples = sidecar.samples
    if len(samples) >= MIN_TREND_SAMPLES:
        span = samples[-1].ts - samples[0].ts
        if span >= MIN_TREND_SPAN_SECONDS:
            oldest_rss = samples[0].rss_bytes
            if oldest_rss > 0:
                change_pct = ((sample.rss_bytes - oldest_rss) / oldest_rss) * 100
                if change_pct > MEM_TREND_GROWTH_PCT:
                    mem_str += f'{RED}\u2191{RESET}'  # ↑ growing
                elif change_pct < -MEM_TREND_SHRINK_PCT:
                    mem_str += f'{GREEN}\u2193{RESET}'  # ↓ shrinking

    # ── Tier 3: Peak memory ──
    # Show peak when it exceeds current by >20% and we have enough samples,
    # indicating a prior spike that has since resolved (useful for leak diagnosis).
    if len(samples) >= MIN_TREND_SAMPLES and sidecar.peak_rss_bytes > sample.rss_bytes * 1.2:
        mem_str += f' {DIM}(peak: {_format_bytes(sidecar.peak_rss_bytes)}){RESET}'

    # ── Tier 1: Tree RSS (shown when children consume significant memory) ──
    # Catches MCP server memory leaks where the main process looks healthy but
    # children are the problem. Shown as parenthetical after main process mem.
    if sample.tree_rss_bytes >= TREE_RSS_WARN_BYTES:
        tree_color = YELLOW if sample.tree_rss_bytes < MEMORY_CRIT_BYTES else RED
        mem_str += f' {tree_color}(tree: {_format_bytes(sample.tree_rss_bytes)}){RESET}'

    parts.append(mem_str)

    # ── Tier 1: CPU (shown after first delta is available) ──
    # Uses persisted samples (not the live sample) because consecutive persisted
    # samples are guaranteed to be ≥MIN_SAMPLE_INTERVAL apart, giving a reliable
    # delta. The live sample may be <0.3s from the last persisted one.
    if len(samples) >= 2:
        cpu = _compute_cpu_percent(samples[-1], samples[-2])
        if cpu is not None:
            cpu_color = _color_for_value(cpu, CPU_WARN_PERCENT, CPU_CRIT_PERCENT)
            parts.append(f'{cpu_color}cpu: {cpu:.0f}%{RESET}')

    # ── Tier 2: Anomaly indicators (conditional) ──
    anomalies: list[str] = []

    if sample.num_zombies > 0:
        label = 'zombie' if sample.num_zombies == 1 else 'zombies'
        anomalies.append(f'{RED}{sample.num_zombies} {label}{RESET}')

    if sample.num_children >= CHILDREN_CRIT_COUNT:
        anomalies.append(f'{RED}children: {sample.num_children}{RESET}')
    elif sample.num_children >= CHILDREN_WARN_COUNT:
        anomalies.append(f'{YELLOW}children: {sample.num_children}{RESET}')

    fd_anomaly_idx: int | None = None
    if sample.num_fds >= FD_WARN_COUNT:
        fd_anomaly_idx = len(anomalies)
        anomalies.append(f'{YELLOW}fds: {sample.num_fds}{RESET}')

    # ── Tier 2: Growth rate alerts ──
    # Catch leaks BEFORE they hit absolute thresholds by detecting sustained
    # growth. A steady +20 FDs/hour signals a leak even at 60 current FDs.
    if len(samples) >= MIN_TREND_SAMPLES:
        span = samples[-1].ts - samples[0].ts
        if span >= MIN_TREND_SPAN_SECONDS:
            span_hours = span / 3600

            fd_growth = (sample.num_fds - samples[0].num_fds) / span_hours
            if fd_growth >= FD_GROWTH_RATE_WARN:
                if fd_anomaly_idx is not None:
                    # Absolute threshold already showing — annotate with rate
                    anomalies[fd_anomaly_idx] = f'{YELLOW}fds: {sample.num_fds} (+{fd_growth:.0f}/hr){RESET}'
                else:
                    anomalies.append(f'{YELLOW}fds: {sample.num_fds} (+{fd_growth:.0f}/hr){RESET}')

            child_growth = (sample.num_children - samples[0].num_children) / span_hours
            if child_growth >= CHILDREN_GROWTH_RATE_WARN:
                if sample.num_children < CHILDREN_WARN_COUNT:
                    anomalies.append(f'{YELLOW}children: {sample.num_children} (+{child_growth:.0f}/hr){RESET}')

    result = ' '.join(parts)
    if anomalies:
        result += f' {DIM}|{RESET} ' + ' '.join(anomalies)
    return result


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    log_path = Path.home() / '.claude-workspace' / 'statusline-error.log'

    try:
        raw = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        msg = f'{datetime.now(UTC).isoformat()}\nJSON decode failed: {e}\n\n'
        with contextlib.suppress(OSError):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(msg)
            log_path.chmod(0o600)
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
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(error_msg)
            log_path.chmod(0o600)

        print(f'{RED}StatusLine: validation failed — see {log_path}{RESET}', file=sys.stderr)
        return

    claude_pid = _find_claude_pid()

    # Collect process health early — before slow operations (keychain, git) —
    # so the CPU delta captures Claude's behavior, not our own overhead.
    health_sample = _collect_health_sample(claude_pid)
    health_sidecar: HealthSidecar | None = None
    if health_sample is not None:
        health_sidecar = _update_health_sidecar(data.session_id, claude_pid, health_sample)

    static, switch_pending = _get_active_credentials(data.session_id, claude_pid)
    branch = _git_branch()
    remote_url = _git_remote_url()

    # ── Line 1: Identity + Model ─────────────────────────────────────────
    parts: list[str] = []

    # Model + Version
    parts.append(f'{CYAN}{data.model.id}{RESET} {DIM}v{data.version}{RESET}')

    # PID + Session ID — space-separated so UUID is double-clickable, links to transcript
    transcript_url = f'file://{data.transcript_path}'
    pid_str = f'{DIM}pid:{RESET} {claude_pid}'
    parts.append(f'{pid_str} {DIM}session_id:{RESET} {_osc8_link(transcript_url, data.session_id)}')

    # Process health (Tier 1: always, Tier 2: anomalies, Tier 3: trend)
    if health_sample is not None and health_sidecar is not None:
        parts.append(_format_health(health_sample, health_sidecar))

    # Account — login ID with compact tier in parens
    is_console = static.billing_type == 'prepaid'

    plan_compact = _format_plan_compact(static.subscription, static.tier)
    if static.auth_method == 'setup-token':
        plan_compact = f'{plan_compact} ⚷'.strip() if plan_compact else '⚷'
    if static.login_name:
        if is_console:
            parts.append(f'{YELLOW}{static.login_name}{RESET}')
        elif plan_compact:
            parts.append(f'{DIM}{static.login_name} ({plan_compact}){RESET}')
        else:
            parts.append(f'{DIM}{static.login_name}{RESET}')
    elif static.email_address:
        parts.append(f'{DIM}{static.email_address}{RESET}')

    if switch_pending:
        parts.append(f'{YELLOW}(switch pending — restart to activate){RESET}')

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
    ctx = data.context_window
    pct = ctx.used_percentage or 0
    metrics.append(_context_bar(pct, ctx.context_window_size))

    # Remaining percentage
    remaining = ctx.remaining_percentage
    if remaining is not None:
        rem_color = GREEN if remaining > 30 else YELLOW if remaining > 10 else RED
        metrics.append(f'{rem_color}{remaining:.0f}% remaining{RESET}')
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

    # ── Line 3: Workspace ──────────────────────────────────────────────
    line3: list[str] = []

    # Show cwd; if project_dir differs, show both
    if data.workspace.project_dir != data.workspace.current_dir:
        line3.append(f'{DIM}cwd:{RESET} {_shorten_path(data.cwd)}')
        line3.append(f'{DIM}project:{RESET} {_shorten_path(data.workspace.project_dir)}')
    else:
        line3.append(f'{DIM}cwd:{RESET} {_shorten_path(data.cwd)}')

    # Transcript path
    line3.append(f'{DIM}transcript:{RESET} {_osc8_link(transcript_url, _shorten_path(data.transcript_path))}')

    print(' │ '.join(line3))

    # ── Line 4: Context Window Detail ───────────────────────────────────
    line4: list[str] = []

    # Cumulative totals
    ctx_in = _format_tokens(ctx.total_input_tokens)
    ctx_out = _format_tokens(ctx.total_output_tokens)
    line4.append(f'{DIM}total_in:{RESET} {ctx_in}')
    line4.append(f'{DIM}total_out:{RESET} {ctx_out}')

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

    # ── Line 5: Debug ───────────────────────────────────────────────────
    mtime = _max_mtime()
    mtime_str = datetime.fromtimestamp(mtime, tz=UTC).astimezone().strftime('%H:%M:%S') if mtime else 'n/a'
    cache_ts = _read_cache_timestamp()
    cache_str = datetime.fromtimestamp(cache_ts, tz=UTC).astimezone().strftime('%H:%M:%S') if cache_ts else 'n/a'
    invalidated = 'yes' if mtime > cache_ts else 'no'
    print(
        f'{DIM}max_mtime:{RESET} {mtime_str} {DIM}│ cache written:{RESET} {cache_str} {DIM}│ invalidated:{RESET} {invalidated}'
    )


def _excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: types.TracebackType | None) -> None:
    if issubclass(exc_type, Exception):
        print(f'{RED}StatusLine crashed: {exc_type.__name__}: {exc_value}{RESET}', file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    else:
        sys.__excepthook__(exc_type, exc_value, exc_tb)


if __name__ == '__main__':
    sys.excepthook = _excepthook
    main()
