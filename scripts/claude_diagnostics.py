#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic>=2.0"]
# ///
# exception_safety_linter.py: skip-file
# strict_typing_linter.py: skip-file

"""
Claude Code installation diagnostics and introspection.

Reads ALL available configuration, authentication, and session data
from a Claude Code installation and presents a comprehensive diagnostic report.

Designed for debugging across different login types, organizations, and configurations.

Data sources:
  1. ~/.claude.json             — Account profile, settings, MCP servers
  2. macOS Keychain             — OAuth tokens, subscription type, rate limit tier
  3. ~/.claude/statsig/         — Feature flag cache (Statsig evaluations)
  4. ~/.claude/projects/        — Session JSONL files
  5. ~/.claude-workspace/       — Hook-tracked sessions (from claude-workspace)
  6. Claude binary              — Version, install path, symlink target

Strictness:
  Sub-objects (oauthAccount, keychain credentials) use Pydantic extra='forbid'.
  If Claude Code updates these JSON shapes, this script fails immediately
  with a clear error showing which fields changed. Update the models to accommodate.

  Top-level ~/.claude.json uses extra='allow' since it changes frequently.
  New/removed top-level keys are reported as informational drift warnings.

Usage:
    ./scripts/claude_diagnostics.py
    ./scripts/claude_diagnostics.py --redact       # Truncate UUIDs/tokens for sharing
    ./scripts/claude_diagnostics.py --json         # Machine-readable JSON output
    ./scripts/claude_diagnostics.py --section auth # Single section only

Exit codes:
    0 — Clean diagnostic (all sections validated)
    1 — Diagnostic completed with validation warnings (schema drift detected)
    2 — Fatal error (can't read essential data sources)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import pydantic

# JSON-serializable datetime: allows string→datetime coercion in strict models.
# Pydantic's strict=True rejects str→datetime by default; this override is the
# standard pattern from claude-session-mcp (JsonDatetime).
type JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]

# =============================================================================
# Configuration
# =============================================================================

CLAUDE_CONFIG_PATH = Path.home() / '.claude.json'
CLAUDE_DIR = Path.home() / '.claude'
STATSIG_DIR = CLAUDE_DIR / 'statsig'
PROJECTS_DIR = CLAUDE_DIR / 'projects'
SESSIONS_JSON = Path.home() / '.claude-workspace' / 'sessions.json'
FALLBACK_CREDENTIALS = CLAUDE_DIR / '.credentials.json'

KEYCHAIN_SERVICE = 'Claude Code-credentials'
MANAGED_SETTINGS = Path('/etc/claude-code/managed-settings.json')
USER_SETTINGS = CLAUDE_DIR / 'settings.json'
USER_SETTINGS_LOCAL = CLAUDE_DIR / 'settings.local.json'

# Settings files in precedence order (highest first).
# Project-level files are resolved relative to cwd at runtime.
SETTINGS_PRECEDENCE: Sequence[tuple[str, Path | None]] = [
    ('Managed (/etc)', MANAGED_SETTINGS),
    ('Project local (.claude/settings.local.json)', None),  # resolved at runtime
    ('Project (.claude/settings.json)', None),  # resolved at runtime
    ('User local (~/.claude/settings.local.json)', USER_SETTINGS_LOCAL),
    ('User (~/.claude/settings.json)', USER_SETTINGS),
    ('Main (~/.claude.json)', CLAUDE_CONFIG_PATH),
]

# =============================================================================
# Known Environment Variables
#
# Grouped by category. Only non-default values are displayed.
# Reference: https://github.com/chrisguillory/claude-session-mcp#readme
# =============================================================================

ENV_VAR_CATEGORIES: Sequence[tuple[str, Sequence[str]]] = [
    (
        'Model Selection',
        [
            'ANTHROPIC_MODEL',
            'CLAUDE_CODE_SUBAGENT_MODEL',
            'ANTHROPIC_DEFAULT_OPUS_MODEL',
            'ANTHROPIC_DEFAULT_SONNET_MODEL',
            'ANTHROPIC_DEFAULT_HAIKU_MODEL',
            'ANTHROPIC_SMALL_FAST_MODEL',
        ],
    ),
    (
        'Output & Thinking',
        [
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS',
            'MAX_THINKING_TOKENS',
            'DISABLE_INTERLEAVED_THINKING',
        ],
    ),
    (
        'Context & Compaction',
        [
            'CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE',
            'DISABLE_AUTO_COMPACT',
            'DISABLE_COMPACT',
            'DISABLE_MICROCOMPACT',
            'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE',
        ],
    ),
    (
        'Caching',
        [
            'DISABLE_PROMPT_CACHING',
            'DISABLE_PROMPT_CACHING_OPUS',
            'DISABLE_PROMPT_CACHING_SONNET',
            'DISABLE_PROMPT_CACHING_HAIKU',
        ],
    ),
    (
        'Limits & Timeouts',
        [
            'MAX_MCP_OUTPUT_TOKENS',
            'MCP_TIMEOUT',
            'MCP_TOOL_TIMEOUT',
            'CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS',
            'BASH_DEFAULT_TIMEOUT_MS',
            'BASH_MAX_TIMEOUT_MS',
            'CLAUDE_CODE_GLOB_TIMEOUT_SECONDS',
            'CLAUDE_CODE_MAX_RETRIES',
            'CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY',
        ],
    ),
    (
        'Privacy & Telemetry',
        [
            'DISABLE_TELEMETRY',
            'DISABLE_ERROR_REPORTING',
            'DISABLE_COST_WARNINGS',
        ],
    ),
    (
        'Other',
        [
            'DISABLE_AUTOUPDATER',
            'CLAUDE_CONFIG_DIR',
            'CLAUDE_ENV_FILE',
            'ANTHROPIC_API_KEY',
            'ENABLE_LSP_TOOL',
            'ENABLE_TOOL_SEARCH',
            'CLAUDE_CODE_ENABLE_TASKS',
            'CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION',
            'CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD',
        ],
    ),
]

DEPRECATED_ENV_VARS: Set[str] = {
    'CLAUDE_CODE_EFFORT_LEVEL',
    'ENABLE_COMMIT_ATTRIBUTION',
}

# Rate limit thresholds by subscription type (from Anthropic docs).
# Values are approximate ITPM (input tokens per minute).
RATE_LIMIT_INFO: Mapping[str, str] = {
    'free': 'Free tier — limited usage',
    'pro': 'Pro — ~40-80 active hours/week (Sonnet)',
    'team': 'Team — shared org capacity, 5x multiplier typical',
    'max': 'Max — up to 20x throughput, ~800+ active hours/week',
    'enterprise': 'Enterprise — custom limits, dedicated capacity',
}

# =============================================================================
# Known Top-Level Keys in ~/.claude.json (for drift detection)
#
# Snapshot from Claude Code 2.1.37. Keys not in this set are reported as new.
# Keys in this set but missing from the file are reported as removed.
# =============================================================================

KNOWN_TOP_LEVEL_KEYS: Set[str] = {
    'anonymousId',
    'appleTerminalBackupPath',
    'appleTerminalSetupInProgress',
    'autoCompactEnabled',
    'autoConnectIde',
    'autoUpdates',
    'autoUpdatesProtectedForNative',
    'bypassPermissionsModeAccepted',
    'cachedChromeExtensionInstalled',
    'cachedDynamicConfigs',
    'cachedGrowthBookFeatures',
    'cachedStatsigGates',
    'changelogLastFetched',
    'claudeCodeFirstTokenDate',
    'claudeInChromeDefaultEnabled',
    'clientDataCache',
    'customApiKeyResponses',
    'fallbackAvailableWarningThreshold',
    'feedbackSurveyState',
    'fileCheckpointingEnabled',
    'firstStartTime',
    'githubRepoPaths',
    'groveConfigCache',
    'hasAcknowledgedCostThreshold',
    'hasAvailableSubscription',
    'hasCompletedClaudeInChromeOnboarding',
    'hasCompletedOnboarding',
    'hasIdeAutoConnectDialogBeenShown',
    'hasIdeOnboardingBeenShown',
    'hasOpusPlanDefault',
    'hasSeenStashHint',
    'hasSeenTasksHint',
    'hasShownOpus45Notice',
    'hasShownOpus46Notice',
    'hasShownS1MWelcomeV2',
    'hasUsedBackslashReturn',
    'hasVisitedPasses',
    'installMethod',
    'isQualifiedForDataSharing',
    'lastOnboardingVersion',
    'lastPlanModeUse',
    'lastReleaseNotesSeen',
    'mcpServers',
    'numStartups',
    'oauthAccount',
    'officialMarketplaceAutoInstallAttempted',
    'officialMarketplaceAutoInstalled',
    'optionAsMetaKeyInstalled',
    'opus45MigrationComplete',
    'opusProMigrationComplete',
    'passesEligibilityCache',
    'passesLastSeenRemaining',
    'passesUpsellSeenCount',
    'penguinModeOrgEnabled',
    'projects',
    'promptQueueUseCount',
    's1mAccessCache',
    's1mNonSubscriberAccessCache',
    'shiftEnterKeyBindingInstalled',
    'showSpinnerTree',
    'skillUsage',
    'sonnet45MigrationComplete',
    'sonnet45MigrationTimestamp',
    'subscriptionNoticeCount',
    'thinkingMigrationComplete',
    'tipsHistory',
    'userID',
    'verbose',
}


# =============================================================================
# Display Mappings
# =============================================================================

SUBSCRIPTION_DISPLAY: Mapping[str, str] = {
    'free': 'Free',
    'pro': 'Claude Pro',
    'team': 'Claude Team',
    'max': 'Claude Max',
    'enterprise': 'Claude Enterprise',
}

BILLING_TYPE_DISPLAY: Mapping[str, str] = {
    'stripe_subscription': 'Stripe Subscription',
}

INSTALL_METHOD_DISPLAY: Mapping[str, str] = {
    'native': 'Native Binary',
    'npm': 'npm Package',
}


# =============================================================================
# ANSI Colors
# =============================================================================


class C:
    """ANSI escape codes for terminal output."""

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def disable(cls) -> None:
        for attr in (
            'RED',
            'GREEN',
            'YELLOW',
            'BLUE',
            'MAGENTA',
            'CYAN',
            'WHITE',
            'BOLD',
            'DIM',
            'RESET',
        ):
            setattr(cls, attr, '')


if not sys.stdout.isatty():
    C.disable()


# =============================================================================
# Strict Pydantic Models — Sub-objects (extra='forbid')
#
# These models fail immediately if Claude Code changes the JSON shape.
# That's the point: schema drift is detected, not silently ignored.
# =============================================================================


class BaseStrictModel(pydantic.BaseModel):
    """Foundation strict model — rejects unknown fields (fail-fast)."""

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )


# -------------------------------------------------------------------------
# OAuth Account (from ~/.claude.json → oauthAccount)
# -------------------------------------------------------------------------

type BillingType = Literal['stripe_subscription']
type OrganizationRole = Literal['owner', 'admin', 'member', 'developer']

# Using str | None for workspaceRole since observed values are limited
# but null is the common case
type WorkspaceRole = str | None


class OAuthAccount(BaseStrictModel):
    """OAuth account profile from ~/.claude.json.

    Strict model — unknown fields cause immediate validation failure.
    This is the identity section of Claude Code's configuration.
    """

    accountUuid: str
    emailAddress: str
    organizationUuid: str
    hasExtraUsageEnabled: bool
    billingType: BillingType
    subscriptionCreatedAt: str  # ISO 8601
    displayName: str
    # Optional fields — present in some accounts, absent in others
    organizationRole: OrganizationRole | None = None
    workspaceRole: WorkspaceRole = None
    organizationName: str | None = None
    # accountCreatedAt observed in personal accounts only
    accountCreatedAt: str | None = None  # ISO 8601


# -------------------------------------------------------------------------
# Keychain Credentials (from macOS Keychain)
# -------------------------------------------------------------------------

type SubscriptionType = Literal['free', 'pro', 'team', 'max', 'enterprise']

type OAuthScope = Literal[
    'user:inference',
    'user:mcp_servers',
    'user:profile',
    'user:sessions:claude_code',
]


class KeychainOAuth(BaseStrictModel):
    """OAuth token data from macOS Keychain.

    Strict model — unknown fields cause immediate validation failure.
    Stored under service 'Claude Code-credentials' in the login keychain.
    """

    accessToken: str
    refreshToken: str
    expiresAt: int  # Unix timestamp in milliseconds
    scopes: Sequence[OAuthScope]
    subscriptionType: SubscriptionType
    rateLimitTier: str  # Too many variants to enumerate as Literal


class KeychainCredentials(BaseStrictModel):
    """Top-level keychain credentials structure."""

    claudeAiOauth: KeychainOAuth


# -------------------------------------------------------------------------
# Statsig Session (from ~/.claude/statsig/)
# -------------------------------------------------------------------------


class StatsigSession(BaseStrictModel):
    """Statsig session tracking data."""

    sessionID: str
    startTime: int  # Unix timestamp ms
    lastUpdate: int  # Unix timestamp ms


# -------------------------------------------------------------------------
# Hook-tracked Sessions (from ~/.claude-workspace/sessions.json)
# -------------------------------------------------------------------------

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']


class SessionMetadata(pydantic.BaseModel):
    """Session metadata from claude-workspace hooks."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)

    claude_pid: int
    process_created_at: JsonDatetime | None = None
    session_ended_at: JsonDatetime | None = None
    session_end_reason: str | None = None
    parent_id: str | None = None
    crash_detected_at: JsonDatetime | None = None
    startup_model: str | None = None
    claude_version: str | None = None


class TrackedSession(pydantic.BaseModel):
    """A session tracked by claude-workspace hooks."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)

    session_id: str
    state: SessionState
    project_dir: str
    transcript_path: str
    source: SessionSource
    metadata: SessionMetadata


class SessionDatabase(pydantic.BaseModel):
    """Container for hook-tracked sessions."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True)

    sessions: Sequence[TrackedSession] = ()


# =============================================================================
# Diagnostic Result Types
# =============================================================================


class BinaryInfo(TypedDict, total=False):
    """Claude binary installation details."""

    binary_path: str
    binary_target: str
    version_from_path: str
    version_string: str


class SessionStats(TypedDict, total=False):
    """Session file statistics from ~/.claude/projects/."""

    project_count: int
    session_file_count: int
    total_size_bytes: int
    error: str


class StatsigData(TypedDict, total=False):
    """Statsig feature flag cache data."""

    stable_id: str
    session: StatsigSession
    cache_last_modified_ms: int
    cache_age: timedelta
    evaluation_top_level_keys: list[str]


class ProcessInfo(TypedDict):
    """A running Claude CLI process."""

    pid: str
    command: str


@dataclass(frozen=True)
class ValidationWarning:
    """A schema drift warning (not fatal, but worth reporting)."""

    section: str
    message: str
    details: str = ''


@dataclass(frozen=True)
class SectionResult:
    """Result of reading one diagnostic section."""

    name: str
    data: Any = None
    error: str | None = None
    warnings: Sequence[ValidationWarning] = ()


@dataclass
class DiagnosticReport:
    """Accumulates all diagnostic sections and warnings."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    sections: list[SectionResult] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# =============================================================================
# Data Readers
# =============================================================================


def read_claude_config() -> tuple[dict[str, Any], Sequence[ValidationWarning]]:
    """Read ~/.claude.json and detect top-level key drift.

    Returns the raw config dict and any drift warnings.
    Raises FileNotFoundError if the file doesn't exist.
    """
    raw = json.loads(CLAUDE_CONFIG_PATH.read_text())
    actual_keys = set(raw.keys())

    warnings: list[ValidationWarning] = []

    new_keys = actual_keys - KNOWN_TOP_LEVEL_KEYS
    if new_keys:
        warnings.append(
            ValidationWarning(
                section='config',
                message=f'New top-level keys in ~/.claude.json ({len(new_keys)})',
                details=', '.join(sorted(new_keys)),
            )
        )

    removed_keys = KNOWN_TOP_LEVEL_KEYS - actual_keys
    if removed_keys:
        warnings.append(
            ValidationWarning(
                section='config',
                message=f'Missing top-level keys from ~/.claude.json ({len(removed_keys)})',
                details=', '.join(sorted(removed_keys)),
            )
        )

    return raw, warnings


def read_oauth_account(config: dict[str, Any]) -> OAuthAccount:
    """Parse oauthAccount from config with strict validation."""
    raw = config.get('oauthAccount')
    if raw is None:
        msg = 'No oauthAccount in ~/.claude.json (not logged in via OAuth?)'
        raise ValueError(msg)
    return OAuthAccount.model_validate(raw)


def read_keychain_credentials() -> KeychainCredentials:
    """Read OAuth credentials from macOS Keychain.

    Raises RuntimeError if keychain is unavailable or entry not found.
    """
    if platform.system() != 'Darwin':
        msg = 'Keychain reading only supported on macOS'
        raise RuntimeError(msg)

    result = subprocess.run(
        [
            'security',
            'find-generic-password',
            '-a',
            os.environ.get('USER', ''),
            '-w',
            '-s',
            KEYCHAIN_SERVICE,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if result.returncode != 0:
        msg = f'Keychain entry not found: service={KEYCHAIN_SERVICE!r}'
        raise RuntimeError(msg)

    raw = json.loads(result.stdout.strip())
    return KeychainCredentials.model_validate(raw)


def read_statsig_data() -> StatsigData:
    """Read Statsig cache data from ~/.claude/statsig/."""
    data = StatsigData()

    if not STATSIG_DIR.exists():
        return data

    # Stable ID
    for f in STATSIG_DIR.glob('statsig.stable_id.*'):
        raw = f.read_text().strip()
        data['stable_id'] = json.loads(raw) if raw.startswith('"') else raw

    # Session
    for f in STATSIG_DIR.glob('statsig.session_id.*'):
        raw = json.loads(f.read_text())
        data['session'] = StatsigSession.model_validate(raw)

    # Last modified time
    for f in STATSIG_DIR.glob('statsig.last_modified_time.*'):
        raw = json.loads(f.read_text())
        # Values are {filename: timestamp_ms}
        for ts_ms in raw.values():
            data['cache_last_modified_ms'] = ts_ms
            data['cache_age'] = datetime.now(UTC) - datetime.fromtimestamp(ts_ms / 1000, tz=UTC)

    # Cached evaluations (just count top-level data keys)
    for f in STATSIG_DIR.glob('statsig.cached.evaluations.*'):
        raw = json.loads(f.read_text())
        data['evaluation_top_level_keys'] = sorted(raw.keys())

    return data


def read_session_stats() -> SessionStats:
    """Count projects and session files in ~/.claude/projects/."""
    if not PROJECTS_DIR.exists():
        return SessionStats(error=f'{PROJECTS_DIR} does not exist')

    project_dirs = [p for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    session_files = list(PROJECTS_DIR.rglob('*.jsonl'))
    total_bytes = sum(f.stat().st_size for f in session_files)

    return SessionStats(
        project_count=len(project_dirs),
        session_file_count=len(session_files),
        total_size_bytes=total_bytes,
    )


def read_tracked_sessions() -> SessionDatabase | None:
    """Read hook-tracked sessions from ~/.claude-workspace/sessions.json."""
    if not SESSIONS_JSON.exists():
        return None
    raw = json.loads(SESSIONS_JSON.read_text())
    return SessionDatabase.model_validate(raw)


def read_mcp_servers(config: dict[str, Any]) -> dict[str, Any]:
    """Extract MCP server configurations from ~/.claude.json."""
    servers: dict[str, Any] = config.get('mcpServers', {})
    return servers


def read_binary_info() -> BinaryInfo:
    """Get Claude binary version and path information."""
    data = BinaryInfo()

    # Find binary
    result = subprocess.run(
        ['which', 'claude'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        binary_path = Path(result.stdout.strip())
        data['binary_path'] = str(binary_path)

        # Resolve symlink
        if binary_path.is_symlink():
            target = binary_path.resolve()
            data['binary_target'] = str(target)
            # Extract version from path (e.g., .../versions/2.1.37)
            if 'versions' in target.parts:
                idx = target.parts.index('versions')
                if idx + 1 < len(target.parts):
                    data['version_from_path'] = target.parts[idx + 1]

    # Get version from CLI
    result = subprocess.run(
        ['claude', '--version'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 0:
        data['version_string'] = result.stdout.strip()

    return data


def read_running_processes() -> Sequence[ProcessInfo]:
    """Find running Claude Code processes (top-level only, not MCP subprocesses)."""
    result = subprocess.run(
        ['pgrep', '-lf', 'claude'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return []

    processes: list[ProcessInfo] = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid, cmd = parts
        # Only top-level claude CLI invocations (not subprocesses)
        first_token = cmd.split()[0]
        is_claude_cli = first_token in ('claude', '/claude') or first_token.endswith('/claude')
        if is_claude_cli:
            processes.append(ProcessInfo(pid=pid, command=cmd))
    return processes


def read_settings_files(cwd: Path) -> Sequence[tuple[str, Path, bool, int]]:
    """Inventory all settings files with existence and key counts.

    Returns list of (label, path, exists, key_count).
    """
    results: list[tuple[str, Path, bool, int]] = []
    for label, static_path in SETTINGS_PRECEDENCE:
        if static_path is None:
            # Project-level: resolve relative to cwd
            if 'local' in label:
                path = cwd / '.claude' / 'settings.local.json'
            else:
                path = cwd / '.claude' / 'settings.json'
        else:
            path = static_path

        exists = path.is_file()
        key_count = 0
        if exists:
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict):
                    key_count = len(data)
            except (json.JSONDecodeError, OSError):
                pass
        results.append((label, path, exists, key_count))
    return results


def read_env_vars(config: dict[str, Any]) -> Sequence[tuple[str, str, str, str]]:
    """Read environment variables from settings.json env block and os.environ.

    Returns list of (var_name, value, source, category) for non-default vars only.
    """
    # Collect env vars from settings.json env blocks
    settings_env: dict[str, tuple[str, str]] = {}  # var -> (value, source)

    for settings_path, source_label in [
        (USER_SETTINGS, '~/.claude/settings.json'),
        (USER_SETTINGS_LOCAL, '~/.claude/settings.local.json'),
    ]:
        if settings_path.is_file():
            try:
                data = json.loads(settings_path.read_text())
                env_block = data.get('env', {})
                if isinstance(env_block, dict):
                    for var, val in env_block.items():
                        settings_env[var] = (str(val), source_label)
            except (json.JSONDecodeError, OSError):
                pass

    # Build lookup: var -> category
    var_to_category: dict[str, str] = {}
    all_known_vars: set[str] = set()
    for category, var_list in ENV_VAR_CATEGORIES:
        for var in var_list:
            var_to_category[var] = category
            all_known_vars.add(var)

    results: list[tuple[str, str, str, str]] = []

    # Settings env vars (highest signal)
    for var, (value, source) in sorted(settings_env.items()):
        category = var_to_category.get(var, 'Custom')
        if var in DEPRECATED_ENV_VARS:
            category = 'Deprecated'
        results.append((var, value, source, category))

    # Shell env vars that are known Claude vars but NOT in settings
    for var in sorted(all_known_vars):
        if var in settings_env:
            continue
        shell_val = os.environ.get(var)
        if shell_val is not None:
            category = var_to_category.get(var, 'Other')
            results.append((var, shell_val, 'shell env', category))

    # Check for deprecated vars in shell env
    for var in sorted(DEPRECATED_ENV_VARS):
        if var in settings_env:
            continue
        shell_val = os.environ.get(var)
        if shell_val is not None:
            results.append((var, shell_val, 'shell env', 'Deprecated'))

    return results


def read_disk_usage() -> Sequence[tuple[str, int, int]]:
    """Read disk usage for ~/.claude/ subdirectories.

    Returns list of (dirname, total_bytes, file_count) for non-empty dirs only.
    """
    if not CLAUDE_DIR.is_dir():
        return []

    results: list[tuple[str, int, int]] = []
    for entry in sorted(CLAUDE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        total_bytes = 0
        file_count = 0
        try:
            for f in entry.rglob('*'):
                if f.is_file():
                    total_bytes += f.stat().st_size
                    file_count += 1
        except OSError:
            pass
        if file_count > 0:
            results.append((entry.name, total_bytes, file_count))
    return results


def check_credential_security() -> tuple[str, str]:
    """Check credential storage mode.

    Returns (mode, detail) where mode is 'keychain', 'plaintext', or 'none'.
    """
    if platform.system() != 'Darwin':
        if FALLBACK_CREDENTIALS.is_file():
            return ('plaintext', str(FALLBACK_CREDENTIALS))
        return ('none', 'No credentials found')

    # Check keychain first
    result = subprocess.run(
        [
            'security',
            'find-generic-password',
            '-a',
            os.environ.get('USER', ''),
            '-s',
            KEYCHAIN_SERVICE,
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    has_keychain = result.returncode == 0

    has_plaintext = FALLBACK_CREDENTIALS.is_file()

    if has_keychain and not has_plaintext:
        return ('keychain', 'macOS Keychain (secure)')
    if has_keychain and has_plaintext:
        return ('keychain+plaintext', f'Keychain + plaintext fallback at {FALLBACK_CREDENTIALS}')
    if has_plaintext:
        return ('plaintext', str(FALLBACK_CREDENTIALS))
    return ('none', 'No credentials found')


# =============================================================================
# Formatting Helpers
# =============================================================================


def fmt_header(title: str) -> str:
    """Format a section header."""
    bar = '─' * 60
    return f'\n{C.BOLD}{C.CYAN}{bar}{C.RESET}\n{C.BOLD}{C.WHITE}  {title}{C.RESET}\n{C.BOLD}{C.CYAN}{bar}{C.RESET}'


def fmt_row(label: str, value: str, indent: int = 2) -> str:
    """Format a label: value row with alignment."""
    pad = ' ' * indent
    return f'{pad}{C.DIM}{label:<26}{C.RESET}{value}'


def fmt_warning(msg: str) -> str:
    """Format a warning message."""
    return f'  {C.YELLOW}⚠ {msg}{C.RESET}'


def fmt_error(msg: str) -> str:
    """Format an error message."""
    return f'  {C.RED}✗ {msg}{C.RESET}'


def fmt_ok(msg: str) -> str:
    """Format a success message."""
    return f'  {C.GREEN}✓ {msg}{C.RESET}'


def fmt_bytes(n: int) -> str:
    """Format byte count as human-readable."""
    if n < 1024:
        return f'{n} B'
    if n < 1024 * 1024:
        return f'{n / 1024:.1f} KB'
    if n < 1024 * 1024 * 1024:
        return f'{n / (1024 * 1024):.1f} MB'
    return f'{n / (1024 * 1024 * 1024):.1f} GB'


def fmt_relative_time(dt: datetime) -> str:
    """Format a datetime relative to now."""
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt
    return _fmt_timedelta(delta) + ' ago'


def fmt_relative_future(dt: datetime) -> str:
    """Format a future datetime relative to now."""
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt - now
    if delta.total_seconds() < 0:
        return f'{C.RED}EXPIRED{C.RESET} ({_fmt_timedelta(-delta)} ago)'
    return f'in {_fmt_timedelta(delta)}'


def _fmt_timedelta(delta: timedelta) -> str:
    """Format a timedelta as human-readable."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f'{total_seconds}s'
    if total_seconds < 3600:
        return f'{total_seconds // 60}m {total_seconds % 60}s'
    hours = total_seconds // 3600
    if hours < 24:
        mins = (total_seconds % 3600) // 60
        return f'{hours}h {mins}m'
    days = hours // 24
    remaining_hours = hours % 24
    return f'{days}d {remaining_hours}h'


def redact_uuid(uuid: str, *, show_chars: int = 8) -> str:
    """Truncate a UUID for sharing."""
    if len(uuid) <= show_chars:
        return uuid
    return uuid[:show_chars] + '…'


def redact_token(token: str) -> str:
    """Redact a token, showing only prefix."""
    if len(token) <= 12:
        return '***'
    return token[:12] + '…***'


# =============================================================================
# Section Renderers
# =============================================================================


def render_identity(
    account: OAuthAccount | None,
    keychain: KeychainCredentials | None,
    config: dict[str, Any],
    *,
    redact: bool,
) -> None:
    """Render the Identity & Authentication section."""
    print(fmt_header('Identity & Authentication'))

    if account is None and keychain is None:
        print(fmt_error('No authentication data found'))
        return

    if account is not None:
        print(fmt_row('Email', f'{C.WHITE}{account.emailAddress}{C.RESET}'))
        print(fmt_row('Display Name', account.displayName))

        org_parts: list[str] = []
        if account.organizationName:
            org_parts.append(f'{C.WHITE}{account.organizationName}{C.RESET}')
        if account.organizationRole:
            org_parts.append(f'({account.organizationRole})')
        if org_parts:
            print(fmt_row('Organization', ' '.join(org_parts)))

        if account.workspaceRole:
            print(fmt_row('Workspace Role', account.workspaceRole))

        billing_display = BILLING_TYPE_DISPLAY.get(account.billingType, account.billingType)
        print(fmt_row('Billing Type', billing_display))

        if redact:
            print(fmt_row('Account UUID', redact_uuid(account.accountUuid)))
            print(fmt_row('Organization UUID', redact_uuid(account.organizationUuid)))
        else:
            print(fmt_row('Account UUID', account.accountUuid))
            print(fmt_row('Organization UUID', account.organizationUuid))

        print(
            fmt_row(
                'Extra Usage',
                f'{C.GREEN}Enabled{C.RESET}' if account.hasExtraUsageEnabled else f'{C.DIM}Disabled{C.RESET}',
            )
        )

        if account.subscriptionCreatedAt:
            try:
                sub_dt = datetime.fromisoformat(account.subscriptionCreatedAt)
                print(fmt_row('Subscription Since', f'{sub_dt:%Y-%m-%d} ({fmt_relative_time(sub_dt)})'))
            except ValueError:
                print(fmt_row('Subscription Since', account.subscriptionCreatedAt))

        if account.accountCreatedAt:
            try:
                acct_dt = datetime.fromisoformat(account.accountCreatedAt)
                print(fmt_row('Account Created', f'{acct_dt:%Y-%m-%d} ({fmt_relative_time(acct_dt)})'))
            except ValueError:
                print(fmt_row('Account Created', account.accountCreatedAt))

    if keychain is not None:
        oauth = keychain.claudeAiOauth
        sub_display = SUBSCRIPTION_DISPLAY.get(oauth.subscriptionType, oauth.subscriptionType)
        print(fmt_row('Subscription', f'{C.BOLD}{C.GREEN}{sub_display}{C.RESET}'))
        print(fmt_row('Rate Limit Tier', oauth.rateLimitTier))
        print(fmt_row('Scopes', ', '.join(oauth.scopes)))

        # Token expiry
        expires_dt = datetime.fromtimestamp(oauth.expiresAt / 1000, tz=UTC)
        remaining = expires_dt - datetime.now(UTC)
        if remaining.total_seconds() < 0:
            expiry_color = C.RED
        elif remaining.total_seconds() < 3600:
            expiry_color = C.YELLOW
        else:
            expiry_color = C.GREEN
        print(
            fmt_row(
                'Token Expires',
                f'{expiry_color}{expires_dt:%Y-%m-%d %H:%M:%S UTC}{C.RESET} ({fmt_relative_future(expires_dt)})',
            )
        )

        if redact:
            print(fmt_row('Access Token', redact_token(oauth.accessToken)))
            print(fmt_row('Refresh Token', redact_token(oauth.refreshToken)))
        else:
            print(fmt_row('Access Token', f'{C.DIM}(present, {len(oauth.accessToken)} chars){C.RESET}'))
            print(fmt_row('Refresh Token', f'{C.DIM}(present, {len(oauth.refreshToken)} chars){C.RESET}'))

    # Derived: what /status would show
    if keychain is not None:
        sub_type = keychain.claudeAiOauth.subscriptionType
        login_display = SUBSCRIPTION_DISPLAY.get(sub_type, sub_type) + ' Account'
    elif account is not None:
        login_display = f'OAuth ({account.billingType})'
    else:
        login_display = 'Unknown'

    print()
    print(fmt_row('/status Login Method', f'{C.BOLD}{login_display}{C.RESET}'))

    # User ID (hashed)
    user_id = config.get('userID', '')
    if user_id:
        if redact:
            print(fmt_row('User ID (hashed)', redact_uuid(user_id, show_chars=12)))
        else:
            print(fmt_row('User ID (hashed)', f'{C.DIM}{user_id}{C.RESET}'))

    anon_id = config.get('anonymousId', '')
    if anon_id:
        if redact:
            print(fmt_row('Anonymous ID', redact_uuid(anon_id, show_chars=16)))
        else:
            print(fmt_row('Anonymous ID', f'{C.DIM}{anon_id}{C.RESET}'))

    # Sonnet 1M access for current org
    if account is not None:
        s1m_cache = config.get('s1mAccessCache', {})
        s1m_entry = s1m_cache.get(account.organizationUuid)
        if s1m_entry is not None:
            has_access = s1m_entry.get('hasAccess', False)
            has_opt_in = s1m_entry.get('hasAccessNotAsDefault', False)
            ts = s1m_entry.get('timestamp')
            age = ''
            if ts:
                age_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                age = f' {C.DIM}(checked {fmt_relative_time(age_dt)}){C.RESET}'
            if has_access:
                display = f'{C.GREEN}✓ Default{C.RESET}{age}'
            elif has_opt_in:
                display = f'{C.GREEN}✓ Available{C.RESET} {C.DIM}(opt-in){C.RESET}{age}'
            else:
                display = f'{C.DIM}✗ Not available{C.RESET}{age}'
            print(fmt_row('Sonnet 1M', display))
        else:
            print(fmt_row('Sonnet 1M', f'{C.DIM}No cached data{C.RESET}'))


def render_installation(
    binary_info: BinaryInfo,
    config: dict[str, Any],
) -> None:
    """Render the Installation section."""
    print(fmt_header('Installation'))

    version = binary_info.get('version_string', 'unknown')
    print(fmt_row('Version', f'{C.WHITE}{version}{C.RESET}'))

    install_method = config.get('installMethod', 'unknown')
    install_display = INSTALL_METHOD_DISPLAY.get(install_method, install_method)
    print(fmt_row('Install Method', install_display))

    if 'binary_path' in binary_info:
        print(fmt_row('Binary Path', binary_info['binary_path']))
    if 'binary_target' in binary_info:
        print(fmt_row('Symlink Target', binary_info['binary_target']))

    print(fmt_row('Platform', f'{platform.system()} {platform.machine()}'))
    print(fmt_row('OS Version', platform.platform()))

    # Usage stats
    num_startups = config.get('numStartups')
    if num_startups is not None:
        print(fmt_row('Total Startups', f'{num_startups:,}'))

    first_start = config.get('firstStartTime')
    if first_start:
        try:
            first_dt = datetime.fromisoformat(first_start)
            print(fmt_row('First Start', f'{first_dt:%Y-%m-%d} ({fmt_relative_time(first_dt)})'))
        except ValueError:
            print(fmt_row('First Start', first_start))

    first_token = config.get('claudeCodeFirstTokenDate')
    if first_token:
        try:
            token_dt = datetime.fromisoformat(first_token)
            print(fmt_row('First Token Date', f'{token_dt:%Y-%m-%d} ({fmt_relative_time(token_dt)})'))
        except ValueError:
            print(fmt_row('First Token Date', first_token))

    onboarding_ver = config.get('lastOnboardingVersion')
    if onboarding_ver:
        print(fmt_row('Onboarding Version', onboarding_ver))


def render_configuration(config: dict[str, Any]) -> None:
    """Render the Configuration section (boolean/scalar settings)."""
    print(fmt_header('Configuration'))

    # Boolean toggles
    toggles: Sequence[tuple[str, str]] = [
        ('autoUpdates', 'Auto Updates'),
        ('autoCompactEnabled', 'Auto Compact'),
        ('fileCheckpointingEnabled', 'File Checkpointing'),
        ('autoConnectIde', 'Auto-connect IDE'),
        ('claudeInChromeDefaultEnabled', 'Claude in Chrome'),
        ('showSpinnerTree', 'Spinner Tree'),
        ('hasOpusPlanDefault', 'Opus Plan Default'),
        ('penguinModeOrgEnabled', 'Penguin Mode (Org)'),
        ('bypassPermissionsModeAccepted', 'Bypass Permissions'),
        ('isQualifiedForDataSharing', 'Data Sharing Qualified'),
    ]

    for key, label in toggles:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            display = f'{C.GREEN}On{C.RESET}' if value else f'{C.DIM}Off{C.RESET}'
        else:
            # Handle 'true'/'false' strings
            display = f'{C.GREEN}On{C.RESET}' if str(value).lower() == 'true' else f'{C.DIM}Off{C.RESET}'
        print(fmt_row(label, display))

    verbose = config.get('verbose')
    if verbose is not None:
        is_verbose = str(verbose).lower() == 'true' if isinstance(verbose, str) else bool(verbose)
        print(fmt_row('Verbose', f'{C.YELLOW}On{C.RESET}' if is_verbose else f'{C.DIM}Off{C.RESET}'))


def render_feature_flags(statsig_data: StatsigData) -> None:
    """Render the Feature Flags (Statsig) section."""
    print(fmt_header('Feature Flags (Statsig)'))

    if not statsig_data:
        print(fmt_warning('No Statsig data found'))
        return

    stable_id = statsig_data.get('stable_id')
    if stable_id:
        print(fmt_row('Stable ID', stable_id))

    session = statsig_data.get('session')
    if isinstance(session, StatsigSession):
        print(fmt_row('Session ID', session.sessionID))
        start_dt = datetime.fromtimestamp(session.startTime / 1000, tz=UTC)
        update_dt = datetime.fromtimestamp(session.lastUpdate / 1000, tz=UTC)
        print(fmt_row('Session Start', f'{start_dt:%Y-%m-%d %H:%M:%S UTC}'))
        print(fmt_row('Last Update', f'{update_dt:%Y-%m-%d %H:%M:%S UTC} ({fmt_relative_time(update_dt)})'))

    cache_age = statsig_data.get('cache_age')
    if isinstance(cache_age, timedelta):
        color = C.YELLOW if cache_age > timedelta(days=1) else C.GREEN
        print(fmt_row('Cache Age', f'{color}{_fmt_timedelta(cache_age)}{C.RESET}'))

    top_keys = statsig_data.get('evaluation_top_level_keys')
    if top_keys:
        print(fmt_row('Cache Structure', ', '.join(top_keys)))


def render_sessions(
    session_stats: SessionStats,
    tracked: SessionDatabase | None,
) -> None:
    """Render the Sessions section."""
    print(fmt_header('Sessions'))

    # File-based stats
    if 'error' in session_stats:
        print(fmt_error(session_stats['error']))
    else:
        print(fmt_row('Projects', f'{session_stats.get("project_count", 0):,}'))
        print(fmt_row('Session Files', f'{session_stats.get("session_file_count", 0):,}'))
        total_bytes = session_stats.get('total_size_bytes', 0)
        print(fmt_row('Total Size', fmt_bytes(total_bytes)))

    # Hook-tracked sessions
    if tracked is None:
        print(fmt_row('Hook Tracking', f'{C.DIM}Not configured (~/.claude-workspace/sessions.json){C.RESET}'))
        return

    sessions = list(tracked.sessions)
    print(fmt_row('Tracked Sessions', f'{len(sessions):,}'))

    # Count by state
    state_counts: dict[str, int] = {}
    for s in sessions:
        state_counts[s.state] = state_counts.get(s.state, 0) + 1

    state_parts: list[str] = []
    for state, count in sorted(state_counts.items()):
        color = {
            'active': C.GREEN,
            'exited': C.DIM,
            'completed': C.BLUE,
            'crashed': C.RED,
        }.get(state, C.WHITE)
        state_parts.append(f'{color}{count} {state}{C.RESET}')
    if state_parts:
        print(fmt_row('By State', ' · '.join(state_parts)))

    # Count by source
    source_counts: dict[str, int] = {}
    for s in sessions:
        source_counts[s.source] = source_counts.get(s.source, 0) + 1
    if source_counts:
        source_parts = [f'{count} {source}' for source, count in sorted(source_counts.items())]
        print(fmt_row('By Source', ' · '.join(source_parts)))


def render_mcp_servers(servers: Mapping[str, Any]) -> None:
    """Render the MCP Servers section."""
    print(fmt_header('MCP Servers'))

    if not servers:
        print(fmt_warning('No MCP servers configured'))
        return

    print(fmt_row('Total', f'{len(servers)} configured'))
    print()

    for name, cfg in sorted(servers.items()):
        server_type = cfg.get('type', '?')
        command = cfg.get('command', '?')
        args = cfg.get('args', [])

        # Type indicator
        type_color = C.CYAN if server_type == 'stdio' else C.MAGENTA
        type_badge = f'{type_color}{server_type}{C.RESET}'

        # Command (truncated if too long)
        cmd_display = command
        if args:
            full_cmd = f'{command} {" ".join(str(a) for a in args)}'
            if len(full_cmd) > 60:
                cmd_display = full_cmd[:57] + '...'
            else:
                cmd_display = full_cmd

        print(f'  {C.WHITE}{name:<32}{C.RESET} [{type_badge}] {C.DIM}{cmd_display}{C.RESET}')


def _build_org_names(config: dict[str, Any]) -> dict[str, str]:
    """Build a mapping of org UUID → display name from all available sources."""
    names: dict[str, str] = {}

    # Current oauthAccount
    oa = config.get('oauthAccount', {})
    org_uuid = oa.get('organizationUuid', '')
    org_name = oa.get('organizationName') or oa.get('emailAddress', '')
    if org_uuid and org_name:
        names[org_uuid] = org_name

    return names


def _fmt_org(uuid: str, names: dict[str, str], current_uuid: str) -> str:
    """Format an org UUID with name resolution and current-org marker."""
    name = names.get(uuid)
    short = f'{uuid[:12]}…'
    marker = f' {C.CYAN}◄{C.RESET}' if uuid == current_uuid else ''
    if name:
        return f'{C.WHITE}{name}{C.RESET} {C.DIM}({short}){C.RESET}{marker}'
    return f'{C.DIM}{short}{C.RESET}{marker}'


def render_access_caches(config: dict[str, Any]) -> None:
    """Render the Access Caches section."""
    print(fmt_header('Access & Caches'))

    org_names = _build_org_names(config)
    current_org = config.get('oauthAccount', {}).get('organizationUuid', '')

    # Sonnet 1M access
    s1m = config.get('s1mAccessCache', {})
    if s1m:
        print(f'  {C.BOLD}Sonnet 1M Access:{C.RESET}')
        for org_uuid, entry in sorted(s1m.items()):
            has_default = entry.get('hasAccess', False)
            has_opt_in = entry.get('hasAccessNotAsDefault', False)
            ts = entry.get('timestamp')
            age = ''
            if ts:
                age_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                age = f' {C.DIM}checked {fmt_relative_time(age_dt)}{C.RESET}'

            if has_default:
                status = f'{C.GREEN}✓ default{C.RESET}'
            elif has_opt_in:
                status = f'{C.GREEN}✓ available{C.RESET} {C.DIM}(opt-in){C.RESET}'
            else:
                status = f'{C.DIM}✗ not available{C.RESET}'

            org_label = _fmt_org(org_uuid, org_names, current_org)
            print(f'    {org_label}')
            print(f'      {status}{age}')

    # Non-subscriber S1M access
    non_sub = config.get('s1mNonSubscriberAccessCache', {})
    if non_sub:
        print(f'  {C.BOLD}Non-Subscriber S1M Access:{C.RESET}')
        for org_uuid, entry in sorted(non_sub.items()):
            has_access = entry.get('hasAccess', False)
            status = f'{C.GREEN}✓ access{C.RESET}' if has_access else f'{C.DIM}✗ no access{C.RESET}'
            org_label = _fmt_org(org_uuid, org_names, current_org)
            print(f'    {org_label}  {status}')

    # Guest passes
    passes = config.get('passesEligibilityCache', {})
    if passes:
        print(f'  {C.BOLD}Guest Passes:{C.RESET}')
        for org_uuid, entry in sorted(passes.items()):
            eligible = entry.get('eligible', False)
            remaining = entry.get('remaining_passes', 0)
            code_details = entry.get('referral_code_details', {})
            code = code_details.get('code', '')
            color = C.GREEN if eligible else C.DIM
            parts = [f'{color}{"✓" if eligible else "✗"} eligible{C.RESET}']
            if remaining:
                parts.append(f'{remaining} remaining')
            if code:
                parts.append(f'code: {code}')
            org_label = _fmt_org(org_uuid, org_names, current_org)
            print(f'    {org_label}  {" · ".join(parts)}')

    if not s1m and not non_sub and not passes:
        print(fmt_row('Caches', f'{C.DIM}No access cache data{C.RESET}'))


def render_processes(processes: Sequence[ProcessInfo]) -> None:
    """Render running Claude processes section."""
    print(fmt_header('Running Processes'))

    if not processes:
        print(fmt_row('Claude Processes', f'{C.DIM}None detected{C.RESET}'))
        return

    print(fmt_row('Claude Processes', f'{len(processes)} running'))
    for proc in processes:
        cmd = proc['command']
        if len(cmd) > 80:
            cmd = cmd[:77] + '...'
        print(f'    {C.DIM}PID {proc["pid"]}{C.RESET}  {cmd}')


def render_settings_files(cwd: Path) -> None:
    """Render the Settings Files Inventory section."""
    print(fmt_header('Settings Files (by precedence)'))

    files = read_settings_files(cwd)
    for label, path, exists, key_count in files:
        if exists:
            # Shorten home directory paths
            display_path = str(path).replace(str(Path.home()), '~')
            keys_info = f'{key_count} keys' if key_count > 0 else 'empty'
            print(f'  {C.GREEN}✓{C.RESET} {label}')
            print(f'    {C.DIM}{display_path} ({keys_info}){C.RESET}')
        else:
            print(f'  {C.DIM}✗ {label}{C.RESET}')


def render_env_vars(config: dict[str, Any]) -> None:
    """Render the Environment Variables section (non-defaults only)."""
    print(fmt_header('Environment Variables'))

    env_vars = read_env_vars(config)

    if not env_vars:
        print(fmt_row('Status', f'{C.DIM}All defaults (no overrides set){C.RESET}'))
        return

    # Group by category
    categories: dict[str, list[tuple[str, str, str]]] = {}
    for var, value, source, category in env_vars:
        categories.setdefault(category, []).append((var, value, source))

    for category, items in categories.items():
        if category == 'Deprecated':
            print(f'  {C.RED}{C.BOLD}{category}:{C.RESET}')
        else:
            print(f'  {C.BOLD}{category}:{C.RESET}')

        for var, value, source in items:
            # Redact API keys
            if 'API_KEY' in var and len(value) > 12:
                display_val = value[:8] + '…***'
            elif len(value) > 50:
                display_val = value[:47] + '...'
            else:
                display_val = value

            deprecation = f' {C.RED}⚠ deprecated{C.RESET}' if var in DEPRECATED_ENV_VARS else ''
            print(f'    {C.WHITE}{var}{C.RESET} = {C.CYAN}{display_val}{C.RESET}')
            print(f'      {C.DIM}from {source}{C.RESET}{deprecation}')

    print()
    print(fmt_row('Total overrides', f'{len(env_vars)}'))


def render_session_cleanup(config: dict[str, Any]) -> None:
    """Render session cleanup policy (cleanupPeriodDays)."""
    # Read from settings files (not ~/.claude.json — it's in settings.json)
    cleanup_days: int | None = None
    cleanup_source = ''

    for settings_path, source_label in [
        (USER_SETTINGS, '~/.claude/settings.json'),
        (USER_SETTINGS_LOCAL, '~/.claude/settings.local.json'),
    ]:
        if settings_path.is_file():
            try:
                data = json.loads(settings_path.read_text())
                if 'cleanupPeriodDays' in data:
                    cleanup_days = int(data['cleanupPeriodDays'])
                    cleanup_source = source_label
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    print(fmt_header('Session Retention'))

    if cleanup_days is None:
        print(fmt_row('cleanupPeriodDays', f'{C.YELLOW}Not set — using default (30 days){C.RESET}'))
        cleanup_days = 30
        cleanup_source = 'default'
    elif cleanup_days >= 9999:
        print(fmt_row('cleanupPeriodDays', f'{C.GREEN}{cleanup_days}{C.RESET} {C.DIM}(effectively never){C.RESET}'))
    elif cleanup_days == 0:
        print(fmt_row('cleanupPeriodDays', f'{C.RED}0 — ALL sessions deleted on startup!{C.RESET}'))
    else:
        print(fmt_row('cleanupPeriodDays', f'{cleanup_days}'))

    if cleanup_source:
        print(fmt_row('Source', f'{C.DIM}{cleanup_source}{C.RESET}'))

    # Show deletion threshold date
    if cleanup_days < 9999:
        threshold = datetime.now(UTC) - timedelta(days=cleanup_days)
        print(fmt_row('Deletion threshold', f'{threshold:%Y-%m-%d} (sessions before this date are deleted on startup)'))


def render_effective_model(config: dict[str, Any], keychain: KeychainCredentials | None) -> None:
    """Render effective model resolution from precedence chain."""
    print(fmt_header('Model Selection'))

    # Check env var sources
    model_var = None
    model_source = ''

    # Check settings.json env blocks
    for settings_path, source_label in [
        (USER_SETTINGS_LOCAL, '~/.claude/settings.local.json'),
        (USER_SETTINGS, '~/.claude/settings.json'),
    ]:
        if settings_path.is_file():
            try:
                data = json.loads(settings_path.read_text())
                env_block = data.get('env', {})
                if isinstance(env_block, dict) and 'ANTHROPIC_MODEL' in env_block:
                    model_var = env_block['ANTHROPIC_MODEL']
                    model_source = f'{source_label} env'
                    break
            except (json.JSONDecodeError, OSError):
                pass

    # Check shell env
    if model_var is None:
        shell_model = os.environ.get('ANTHROPIC_MODEL')
        if shell_model:
            model_var = shell_model
            model_source = 'shell env'

    # Subscription default
    sub_default = 'sonnet'
    if keychain is not None:
        sub_type = keychain.claudeAiOauth.subscriptionType
        if sub_type in ('max', 'enterprise'):
            sub_default = 'sonnet'  # Max defaults to sonnet unless overridden
        if config.get('hasOpusPlanDefault'):
            sub_default = 'opus'

    if model_var:
        print(fmt_row('Effective model', f'{C.WHITE}{model_var}{C.RESET}'))
        print(fmt_row('Source', f'{C.DIM}{model_source}{C.RESET}'))
    else:
        print(fmt_row('Effective model', f'{C.DIM}{sub_default} (subscription default){C.RESET}'))

    print(
        fmt_row(
            'Opus plan default', f'{C.GREEN}Yes{C.RESET}' if config.get('hasOpusPlanDefault') else f'{C.DIM}No{C.RESET}'
        )
    )

    # Subagent model
    subagent_model = None
    for settings_path in [USER_SETTINGS_LOCAL, USER_SETTINGS]:
        if settings_path.is_file():
            try:
                data = json.loads(settings_path.read_text())
                env_block = data.get('env', {})
                if isinstance(env_block, dict) and 'CLAUDE_CODE_SUBAGENT_MODEL' in env_block:
                    subagent_model = env_block['CLAUDE_CODE_SUBAGENT_MODEL']
                    break
            except (json.JSONDecodeError, OSError):
                pass
    if subagent_model is None:
        subagent_model = os.environ.get('CLAUDE_CODE_SUBAGENT_MODEL')

    if subagent_model:
        print(fmt_row('Subagent model', f'{C.WHITE}{subagent_model}{C.RESET}'))
    else:
        print(fmt_row('Subagent model', f'{C.DIM}(default per agent type){C.RESET}'))


def render_rate_limits(keychain: KeychainCredentials | None) -> None:
    """Render rate limit information derived from subscription type."""
    print(fmt_header('Rate Limits'))

    if keychain is None:
        print(fmt_warning('No keychain data — cannot determine rate limit tier'))
        return

    oauth = keychain.claudeAiOauth
    sub_display = SUBSCRIPTION_DISPLAY.get(oauth.subscriptionType, oauth.subscriptionType)
    tier_info = RATE_LIMIT_INFO.get(oauth.subscriptionType, 'Unknown tier')

    print(fmt_row('Subscription', f'{C.WHITE}{sub_display}{C.RESET}'))
    print(fmt_row('Rate Limit Tier', oauth.rateLimitTier))
    print(fmt_row('Tier Description', f'{C.DIM}{tier_info}{C.RESET}'))


def render_credential_security() -> None:
    """Render credential storage security assessment."""
    print(fmt_header('Credential Security'))

    mode, detail = check_credential_security()

    if mode == 'keychain':
        print(fmt_ok(detail))
    elif mode == 'plaintext':
        print(fmt_error(f'Credentials in PLAINTEXT: {detail}'))
        print(fmt_warning('Move credentials to macOS Keychain for security'))
    elif mode == 'keychain+plaintext':
        print(fmt_warning(detail))
        print(fmt_warning('Plaintext fallback exists alongside keychain — consider removing'))
    else:
        print(fmt_row('Status', f'{C.DIM}{detail}{C.RESET}'))


def render_disk_usage() -> None:
    """Render disk usage for ~/.claude/ (anomalies only)."""
    print(fmt_header('Disk Usage (~/.claude/)'))

    usage = read_disk_usage()
    if not usage:
        print(fmt_row('Status', f'{C.DIM}~/.claude/ not found{C.RESET}'))
        return

    total_bytes = sum(b for _, b, _ in usage)
    total_files = sum(c for _, _, c in usage)
    print(fmt_row('Total', f'{fmt_bytes(total_bytes)} across {total_files:,} files'))
    print()

    # Thresholds for flagging
    thresholds = {
        'projects': 500 * 1024 * 1024,  # 500 MB
        'debug': 100 * 1024 * 1024,  # 100 MB
        'file-history': 500 * 1024 * 1024,  # 500 MB
    }

    for dirname, dir_bytes, file_count in usage:
        threshold = thresholds.get(dirname, 50 * 1024 * 1024)  # 50 MB default
        if dir_bytes >= threshold:
            color = C.YELLOW
            flag = ' ⚠'
        else:
            color = C.DIM
            flag = ''

        print(f'  {color}{dirname + "/":<24}{fmt_bytes(dir_bytes):>10}  ({file_count:,} files){flag}{C.RESET}')


def render_drift_warnings(warnings: Sequence[ValidationWarning]) -> None:
    """Render schema drift warnings at the end."""
    if not warnings:
        return

    print(fmt_header('Schema Drift Warnings'))
    for w in warnings:
        print(fmt_warning(f'[{w.section}] {w.message}'))
        if w.details:
            print(f'    {C.DIM}{w.details}{C.RESET}')


def render_validation_error(section: str, error: pydantic.ValidationError, raw: Any) -> None:
    """Render a Pydantic validation error with the raw data for debugging."""
    print(fmt_error(f'{section}: Pydantic validation failed (schema drift detected)'))
    print()
    for err in error.errors():
        loc = ' → '.join(str(x) for x in err['loc'])
        print(f'    {C.RED}Field:{C.RESET} {loc}')
        print(f'    {C.RED}Error:{C.RESET} {err["msg"]}')
        print(f'    {C.RED}Type:{C.RESET}  {err["type"]}')
        if 'input' in err:
            input_str = str(err['input'])
            if len(input_str) > 100:
                input_str = input_str[:97] + '...'
            print(f'    {C.RED}Input:{C.RESET}  {input_str}')
        print()

    # Show raw JSON for manual inspection
    if raw is not None:
        print(f'    {C.DIM}Raw JSON:{C.RESET}')
        raw_str = json.dumps(raw, indent=2, default=str)
        for line in raw_str.split('\n')[:20]:
            print(f'    {C.DIM}{line}{C.RESET}')
        lines = raw_str.count('\n')
        if lines > 20:
            print(f'    {C.DIM}... ({lines - 20} more lines){C.RESET}')


# =============================================================================
# JSON Output Mode
# =============================================================================


def build_json_report(
    config: dict[str, Any],
    account: OAuthAccount | None,
    keychain: KeychainCredentials | None,
    binary_info: BinaryInfo,
    statsig_data: StatsigData,
    session_stats: SessionStats,
    tracked: SessionDatabase | None,
    warnings: Sequence[ValidationWarning],
    *,
    redact: bool,
) -> dict[str, Any]:
    """Build machine-readable JSON report."""
    report: dict[str, Any] = {
        'timestamp': datetime.now(UTC).isoformat(),
        'identity': {},
        'installation': binary_info,
        'configuration': {},
        'sessions': session_stats,
        'warnings': [{'section': w.section, 'message': w.message, 'details': w.details} for w in warnings],
    }

    if account is not None:
        acct_data = account.model_dump()
        if redact:
            acct_data['accountUuid'] = redact_uuid(acct_data['accountUuid'])
            acct_data['organizationUuid'] = redact_uuid(acct_data['organizationUuid'])
        report['identity']['account'] = acct_data

    if keychain is not None:
        kc_data = keychain.model_dump()
        kc_data['claudeAiOauth']['accessToken'] = '***'
        kc_data['claudeAiOauth']['refreshToken'] = '***'
        report['identity']['keychain'] = kc_data

    if tracked is not None:
        state_counts: dict[str, int] = {}
        for s in tracked.sessions:
            state_counts[s.state] = state_counts.get(s.state, 0) + 1
        report['sessions']['tracked'] = {
            'total': len(tracked.sessions),
            'by_state': state_counts,
        }

    return report


# =============================================================================
# Main
# =============================================================================

SECTIONS = (
    'auth',
    'install',
    'model',
    'config',
    'env',
    'settings',
    'retention',
    'limits',
    'security',
    'flags',
    'sessions',
    'mcp',
    'access',
    'disk',
    'processes',
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Claude Code installation diagnostics and introspection.',
        epilog='Reads all available config, auth, and session data for debugging.',
    )
    parser.add_argument(
        '--redact',
        action='store_true',
        help='Truncate UUIDs and tokens for sharing',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        dest='json_output',
        help='Output machine-readable JSON',
    )
    parser.add_argument(
        '--section',
        choices=SECTIONS,
        default=None,
        help='Show only a specific section',
    )
    args = parser.parse_args()

    all_warnings: list[ValidationWarning] = []
    has_errors = False

    # ── Read config ──────────────────────────────────────────────────────
    try:
        config, config_warnings = read_claude_config()
        all_warnings.extend(config_warnings)
    except FileNotFoundError:
        print(fmt_error(f'Cannot read {CLAUDE_CONFIG_PATH} — is Claude Code installed?'))
        return 2
    except json.JSONDecodeError as e:
        print(fmt_error(f'Cannot parse {CLAUDE_CONFIG_PATH}: {e}'))
        return 2

    # ── Read sub-objects ─────────────────────────────────────────────────
    account: OAuthAccount | None = None
    try:
        account = read_oauth_account(config)
    except (ValueError, pydantic.ValidationError) as e:
        if isinstance(e, pydantic.ValidationError):
            render_validation_error('oauthAccount', e, config.get('oauthAccount'))
            has_errors = True
        else:
            all_warnings.append(ValidationWarning(section='auth', message=str(e)))

    keychain: KeychainCredentials | None = None
    try:
        keychain = read_keychain_credentials()
    except (RuntimeError, json.JSONDecodeError, pydantic.ValidationError) as e:
        if isinstance(e, pydantic.ValidationError):
            render_validation_error('keychain', e, None)
            has_errors = True
        else:
            all_warnings.append(ValidationWarning(section='auth', message=f'Keychain: {e}'))

    binary_info = read_binary_info()
    statsig_data = read_statsig_data()
    session_stats = read_session_stats()

    tracked: SessionDatabase | None = None
    try:
        tracked = read_tracked_sessions()
    except pydantic.ValidationError as e:
        render_validation_error('sessions.json', e, None)
        has_errors = True

    mcp_servers = read_mcp_servers(config)
    processes = read_running_processes()

    # ── JSON output mode ─────────────────────────────────────────────────
    if args.json_output:
        report = build_json_report(
            config,
            account,
            keychain,
            binary_info,
            statsig_data,
            session_stats,
            tracked,
            all_warnings,
            redact=args.redact,
        )
        print(json.dumps(report, indent=2, default=str))
        return 1 if has_errors or all_warnings else 0

    # ── Banner ───────────────────────────────────────────────────────────
    now = datetime.now(tz=UTC)
    banner = f'{C.BOLD}{C.WHITE}Claude Code Diagnostics{C.RESET} — {now:%Y-%m-%d %H:%M:%S %Z}'
    bar = '═' * 60
    print(f'\n{C.CYAN}{bar}{C.RESET}')
    print(f'  {banner}')
    print(f'{C.CYAN}{bar}{C.RESET}')

    # ── Render sections ──────────────────────────────────────────────────
    show = args.section

    if show is None or show == 'auth':
        render_identity(account, keychain, config, redact=args.redact)

    if show is None or show == 'install':
        render_installation(binary_info, config)

    if show is None or show == 'model':
        render_effective_model(config, keychain)

    if show is None or show == 'config':
        render_configuration(config)

    if show is None or show == 'env':
        render_env_vars(config)

    if show is None or show == 'settings':
        render_settings_files(Path.cwd())

    if show is None or show == 'retention':
        render_session_cleanup(config)

    if show is None or show == 'limits':
        render_rate_limits(keychain)

    if show is None or show == 'security':
        render_credential_security()

    if show is None or show == 'flags':
        render_feature_flags(statsig_data)

    if show is None or show == 'sessions':
        render_sessions(session_stats, tracked)

    if show is None or show == 'mcp':
        render_mcp_servers(mcp_servers)

    if show is None or show == 'access':
        render_access_caches(config)

    if show is None or show == 'disk':
        render_disk_usage()

    if show is None or show == 'processes':
        render_processes(processes)

    # ── Drift warnings ───────────────────────────────────────────────────
    if show is None:
        render_drift_warnings(all_warnings)

    # ── Footer ───────────────────────────────────────────────────────────
    if show is None:
        print()
        if has_errors:
            print(f'{C.RED}⚠ Schema drift detected — update models to accommodate{C.RESET}')
        elif all_warnings:
            print(f'{C.YELLOW}ℹ {len(all_warnings)} drift warning(s) — see above{C.RESET}')
        else:
            print(f'{C.GREEN}✓ All sections validated cleanly{C.RESET}')
        print()

    return 1 if has_errors or all_warnings else 0


if __name__ == '__main__':
    sys.exit(main())
