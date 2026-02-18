#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic>=2.0", "psutil>=5.9", "typer>=0.9.0", "local_lib"]
#
# [tool.uv.sources]
# local_lib = { path = "../local-lib/", editable = true }
# ///

"""Claude Code login and MCP auth manager.

Manages multiple Claude accounts and MCP server OAuth tokens across account
switches. Claude Code uses two keychain entries for different auth types:

    "Claude Code"              API key (Console/prepaid accounts)
    "Claude Code-credentials"  OAuth tokens + MCP tokens (subscription accounts)

This tool manages both, enforcing mutual exclusivity when switching between
account types. It also preserves MCP server tokens that Claude Code's /login
flow would otherwise destroy.

Saved login management:
    save-login                 Save current Claude auth (auto-named)
    switch-login <id>          Switch to saved login + inject MCP tokens
    list-logins                List all saved logins
    delete-login <id>          Delete a saved login

Setup token management:
    save-setup-token --login <id>  Save long-lived token (prompted securely)
    clear-setup-token <id>         Remove setup token from login

Saved MCP login management:
    save-mcp-login <server>    Save MCP server token from keychain
    list-mcp-logins            List saved MCP logins
    delete-mcp-login <server>  Remove saved MCP login

Claude Code auth state:
    current-claude-login       Show active/pending auth state
    nuke-claude-auth           Wipe all Claude Code auth state

Workflow:
    1. /login to each account, then save-login for each
    2. For MCP servers: /mcp to auth, then save-mcp-login for each
    3. switch-login <id> to switch accounts (injects MCP tokens automatically)
    4. Restart Claude Code to activate

Setup token workflow (subscription accounts only):
    1. Run `claude setup-token` (opens browser — choose any account)
    2. save-setup-token --login <id> (pastes token to specific login)
    3. switch-login <id> writes setup-token to keychain as accessToken
    4. switch-login <id> --keychain to use original OAuth credentials instead

Login IDs are auto-derived: email--{Console,Team,Personal}

Run directly from GitHub (no clone needed):
    uv run https://raw.githubusercontent.com/chrisguillory/claude-workspace/main/scripts/claude-login.py list-logins

Pin to a specific commit for deterministic behavior:
    uv run https://raw.githubusercontent.com/chrisguillory/claude-workspace/<sha>/scripts/claude-login.py list-logins

Tab completion (requires symlink, not remote uv run):
    ln -s ~/claude-workspace/scripts/claude-login.py ~/.local/bin/claude-login
    claude-login completion zsh --install
    # Or print to stdout: claude-login completion zsh > "$ZDOTDIR/completions/_claude-login"
    # Restart shell, then:
    claude-login switch-login <TAB>

    Note: typer derives the command name from sys.argv[0], so a symlink with a clean
    name is required.

Claude Code CLI auth commands (for reference):
    claude auth login [--email <email>] [--sso]   Non-interactive OAuth login
    claude auth status [--json|--text]             Check auth state (exit 0=ok, 1=no)
    claude auth logout                             Clear all auth state
    claude setup-token                             Browser OAuth → 1-year token (stdout)
    /login                                         Interactive OAuth (inside REPL)
    /logout                                        Clear auth (inside REPL)

    Setup tokens (sk-ant-oat01-*) have user:inference scope only — /status and
    `claude usage` won't show profile info. `claude setup-token` opens the
    same browser OAuth flow as /login — you choose the account there,
    independent of which account is currently active. No way to skip browser.

    The settings.json `env` block has a security allowlist that excludes all
    credential env vars (CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_API_KEY, etc.).
    Credentials must come from shell environment or OS keychain.

Token format reference:
    sk-ant-oat01-*   OAuth Access Token v1 (both /login and setup-token)
    sk-ant-api03-*   Console API key

    OAuth and setup tokens are visually identical (same prefix, 108 chars).
    The only distinction is server-side: setup tokens have 1-year expiry and
    user:inference scope only. Locally, setup tokens are stored with
    refreshToken: null in the keychain — that's how we differentiate them.

Possible improvements:
- Keychain `-T` flag: pass `-T /path/to/claude` to explicitly grant access.
  Not currently needed (no ACL restrictions observed). See:
  https://macromates.com/blog/2006/keychain-access-from-shell/
- Cross-platform: Linux uses ~/.claude/.credentials.json instead of Keychain.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import pydantic
import rich.console
import rich.panel
import typer
from local_lib.schemas import StrictModel
from local_lib.session_tracker import find_claude_pid, resolve_session_id

# =============================================================================
# Pydantic Models
# =============================================================================


class PermissiveModel(pydantic.BaseModel):
    """For Claude Code data — allow unknown fields for forward compatibility."""

    model_config = pydantic.ConfigDict(extra='allow', frozen=True)


class OAuthAccount(PermissiveModel):
    """Claude Code oauthAccount from ~/.claude.json."""

    # Always present after login
    account_uuid: str
    email_address: str
    organization_uuid: str

    # Populated by profile refresh (may be missing on fresh login)
    billing_type: str | None = None
    has_extra_usage_enabled: bool | None = None
    display_name: str | None = None
    subscription_created_at: str | None = None
    organization_name: str | None = None
    organization_role: str | None = None
    workspace_role: str | None = None
    account_created_at: str | None = None


class ClaudeAiOAuth(PermissiveModel):
    """Claude account OAuth tokens from keychain.

    Fields are nullable because setup-tokens write accessToken with all other
    fields as null. save-login must handle reading this state from keychain.
    """

    access_token: str
    refresh_token: str | None = None
    expires_at: int | None = None
    scopes: Sequence[str] = ()
    subscription_type: str | None = None
    rate_limit_tier: str | None = None


class McpOAuthEntry(PermissiveModel):
    """MCP server OAuth token entry."""

    server_name: str
    server_url: str
    client_id: str
    access_token: str
    expires_at: int
    refresh_token: str
    scope: str
    keychain_key: str  # Original compound key (e.g., "linear|638130d5ab3558f4")


class LoginFile(StrictModel):
    """Saved login credentials.

    Optional fields have defaults so pre-migration files and tab completion
    (which bypasses @app.callback) can deserialize without crashing.
    """

    name: str
    created_at: datetime
    oauth_account: OAuthAccount
    claude_ai_oauth: ClaudeAiOAuth | None = None
    api_key: str | None = None
    setup_token: str | None = None
    setup_token_created_at: datetime | None = None


# =============================================================================
# Constants
# =============================================================================

CONFIG_PATH = Path.home() / '.claude.json'
WORKSPACE_DIR = Path.home() / '.claude-workspace'
LOGINS_DIR = WORKSPACE_DIR / 'logins'
MCP_AUTHS_PATH = WORKSPACE_DIR / 'mcp-auths.json'
SWITCH_PENDING_PATH = WORKSPACE_DIR / '.switch-pending'
KEYCHAIN_SERVICE_CREDENTIALS = 'Claude Code-credentials'
KEYCHAIN_SERVICE_API_KEY = 'Claude Code'
SETUP_TOKEN_PREFIX = 'sk-ant-oat01-'
SETUP_TOKEN_LIFETIME_DAYS = 365


# =============================================================================
# Process Relaunch
# =============================================================================

_KILL_SCRIPT = """\
import os, signal, time
pid = {claude_pid}
time.sleep(0.5)
os.kill(pid, signal.SIGTERM)
for _ in range(50):
    time.sleep(0.1)
    try:
        os.kill(pid, 0)
    except OSError:
        break
"""


def _spawn_kill_and_copy_resume(claude_pid: int, session_id: str) -> None:
    """Spawn a detached process that kills Claude, and copy the resume command.

    The detached process survives the parent's death (start_new_session=True).
    Sequence: sleep 0.5s → SIGTERM → wait up to 5s for exit.

    The resume command is copied to clipboard so the user can Cmd+V + Enter
    after Claude exits. A detached process cannot launch a TUI app in the
    user's terminal, so we rely on the clipboard instead.
    """
    resume_cmd = f'claude --resume {session_id}'
    subprocess.run(['pbcopy'], input=resume_cmd.encode(), check=False)

    script = _KILL_SCRIPT.format(claude_pid=claude_pid)
    subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


# =============================================================================
# Keychain Operations
# =============================================================================


def read_keychain_raw() -> Mapping[str, Any] | None:  # strict_typing_linter.py: loose-typing
    """Read full keychain JSON. Returns None if entry doesn't exist."""
    result = subprocess.run(
        ['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE_CREDENTIALS, '-w'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    data: Mapping[str, Any] = json.loads(result.stdout.strip())
    return data


def write_keychain_raw(data: Mapping[str, Any]) -> None:  # strict_typing_linter.py: loose-typing
    """Write full keychain JSON. Deletes then re-creates the entry."""
    user = os.environ.get('USER', '')
    # Delete ALL existing entries with this service (may be multiple with different -a values)
    for _ in range(5):
        del_result = subprocess.run(
            ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE_CREDENTIALS],
            capture_output=True,
            timeout=5,
        )
        if del_result.returncode != 0:
            break
    # Write new entry
    result = subprocess.run(
        ['security', 'add-generic-password', '-s', KEYCHAIN_SERVICE_CREDENTIALS, '-a', user, '-w', json.dumps(data)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        print(f'ERROR writing keychain: {result.stderr.strip()}', file=sys.stderr)
        sys.exit(1)


def delete_keychain() -> None:
    """Delete all keychain entries for credentials service."""
    for _ in range(5):
        result = subprocess.run(
            ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE_CREDENTIALS],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            break


def read_api_key_keychain() -> str | None:
    """Read raw API key from 'Claude Code' keychain entry."""
    result = subprocess.run(
        ['security', 'find-generic-password', '-s', KEYCHAIN_SERVICE_API_KEY, '-w'],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if not value or value.startswith('{'):
        return None
    return value


def write_api_key_keychain(api_key: str) -> None:
    """Write raw API key to 'Claude Code' keychain (hex-encoded)."""
    user = os.environ.get('USER', '')
    # Delete existing
    subprocess.run(
        ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE_API_KEY],
        capture_output=True,
        timeout=5,
    )
    # Hex encoding matches Claude Code's W49: Buffer.from(key, "utf-8").toString("hex")
    hex_key = api_key.encode('utf-8').hex()
    result = subprocess.run(
        ['security', 'add-generic-password', '-U', '-a', user, '-s', KEYCHAIN_SERVICE_API_KEY, '-X', hex_key],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        print(f'ERROR writing API key keychain: {result.stderr.strip()}', file=sys.stderr)
        sys.exit(1)


def delete_api_key_keychain() -> None:
    """Delete 'Claude Code' keychain entry (API key)."""
    subprocess.run(
        ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE_API_KEY],
        capture_output=True,
        timeout=5,
    )


# =============================================================================
# Config Operations
# =============================================================================


_OAUTH_CAMEL_TO_SNAKE: Mapping[str, str] = {
    'accountUuid': 'account_uuid',
    'emailAddress': 'email_address',
    'organizationUuid': 'organization_uuid',
    'billingType': 'billing_type',
    'hasExtraUsageEnabled': 'has_extra_usage_enabled',
    'displayName': 'display_name',
    'subscriptionCreatedAt': 'subscription_created_at',
    'organizationName': 'organization_name',
    'organizationRole': 'organization_role',
    'workspaceRole': 'workspace_role',
    'accountCreatedAt': 'account_created_at',
}


def read_oauth_account() -> OAuthAccount | None:
    """Read oauthAccount from ~/.claude.json."""
    if not CONFIG_PATH.is_file():
        return None
    config = json.loads(CONFIG_PATH.read_text())
    raw = config.get('oauthAccount')
    if not raw:
        return None
    # Convert known camelCase keys to snake_case; pass unknown keys through
    # so extra='allow' on PermissiveModel preserves forward-compat fields.
    converted: dict[str, Any] = {}
    for camel_key, value in raw.items():
        snake_key = _OAUTH_CAMEL_TO_SNAKE.get(camel_key)
        converted[snake_key if snake_key else camel_key] = value
    return OAuthAccount.model_validate(converted)


_OAUTH_SNAKE_TO_CAMEL: Mapping[str, str] = {v: k for k, v in _OAUTH_CAMEL_TO_SNAKE.items()}


def write_oauth_account(account: OAuthAccount) -> None:
    """Write oauthAccount to ~/.claude.json, preserving all other fields."""
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
    dumped = account.model_dump()
    camel: dict[str, Any] = {}
    # Convert known snake_case fields back to camelCase
    for snake_key, value in dumped.items():
        camel_key = _OAUTH_SNAKE_TO_CAMEL.get(snake_key)
        if camel_key:
            # Skip None optional fields to avoid breaking Claude Code
            if value is not None or snake_key in ('account_uuid', 'email_address', 'organization_uuid'):
                camel[camel_key] = value
        else:
            # Unknown extras (forward-compat) — pass through as-is
            camel[snake_key] = value

    config['oauthAccount'] = camel
    _write_file_atomically(CONFIG_PATH, json.dumps(config, indent=2))


def remove_oauth_account() -> None:
    """Remove oauthAccount from ~/.claude.json."""
    if not CONFIG_PATH.is_file():
        return
    config = json.loads(CONFIG_PATH.read_text())
    config.pop('oauthAccount', None)
    _write_file_atomically(CONFIG_PATH, json.dumps(config, indent=2))


def remove_primary_api_key() -> None:
    """Remove primaryApiKey from ~/.claude.json if present.

    Claude Code stores this as a fallback when keychain write fails.
    Must be cleaned up when switching to subscription accounts.
    """
    if not CONFIG_PATH.is_file():
        return
    config = json.loads(CONFIG_PATH.read_text())
    if 'primaryApiKey' in config:
        del config['primaryApiKey']
        _write_file_atomically(CONFIG_PATH, json.dumps(config, indent=2))


# =============================================================================
# File Operations
# =============================================================================


def _write_file_atomically(path: Path, content: str) -> None:
    """Write content to a temp file, fsync, then atomic rename."""
    tmp = path.with_suffix('.tmp')
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, content.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# =============================================================================
# Login File Migration
# =============================================================================

# Fields added after initial LoginFile schema. Each entry is (field_name, default_value).
_LOGIN_MIGRATION_FIELDS: Sequence[tuple[str, Any]] = [  # strict_typing_linter.py: loose-typing
    ('api_key', None),
    ('setup_token', None),
    ('setup_token_created_at', None),
]


def migrate_login_files() -> int:
    """Add missing fields to login files. Returns count migrated."""
    if not LOGINS_DIR.exists():
        return 0
    migrated = 0
    for path in sorted(LOGINS_DIR.glob('*.json')):
        raw: dict[str, Any] = json.loads(path.read_text())
        changed = False
        for field, default in _LOGIN_MIGRATION_FIELDS:
            if field not in raw:
                raw[field] = default
                changed = True
        if changed:
            _write_file_atomically(path, json.dumps(raw, indent=2))
            migrated += 1
    return migrated


# =============================================================================
# Name Derivation
# =============================================================================


def derive_login_name(account: OAuthAccount, oauth: ClaudeAiOAuth | None) -> str:
    """Derive login ID from account data."""
    email = account.email_address

    if account.billing_type == 'prepaid':
        return f'{email}--Console'

    if oauth and oauth.subscription_type == 'team':
        return f'{email}--Team'

    return f'{email}--Personal'


# =============================================================================
# Login Storage
# =============================================================================


def login_path(name: str) -> Path:
    """Return path for a login file."""
    return LOGINS_DIR / f'{name}.json'


def save_login(login: LoginFile) -> Path:
    """Save login to disk."""
    LOGINS_DIR.mkdir(parents=True, exist_ok=True)
    path = login_path(login.name)
    _write_file_atomically(path, login.model_dump_json(indent=2))
    return path


def load_login(name: str) -> LoginFile:
    """Load login by exact name."""
    path = login_path(name)
    if not path.exists():
        print(f'Login not found: {name}', file=sys.stderr)
        sys.exit(1)
    return LoginFile.model_validate_json(path.read_text())


def list_all_logins() -> Sequence[LoginFile]:
    """List all saved logins."""
    if not LOGINS_DIR.exists():
        return []
    return [LoginFile.model_validate_json(path.read_text()) for path in sorted(LOGINS_DIR.glob('*.json'))]


def delete_login_file(name: str) -> None:
    """Delete login file."""
    path = login_path(name)
    if path.exists():
        path.unlink()


# =============================================================================
# MCP Auth Storage
# =============================================================================


def save_mcp_auths(auths: Mapping[str, McpOAuthEntry]) -> None:
    """Save MCP auths to disk."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    serialized = {k: v.model_dump() for k, v in auths.items()}
    _write_file_atomically(MCP_AUTHS_PATH, json.dumps(serialized, indent=2))


def load_mcp_auths() -> Mapping[str, McpOAuthEntry]:
    """Load MCP auths from disk."""
    if not MCP_AUTHS_PATH.exists():
        return {}
    raw = json.loads(MCP_AUTHS_PATH.read_text())
    return {k: McpOAuthEntry.model_validate(v) for k, v in raw.items()}


def mcp_auths_to_keychain(  # strict_typing_linter.py: loose-typing
    auths: Mapping[str, McpOAuthEntry],
) -> Mapping[str, Any]:
    """Convert MCP auths to keychain format."""
    result = {}
    for entry in auths.values():
        data = entry.model_dump()
        kc_key = data.pop('keychain_key')
        # Convert snake_case back to camelCase for keychain
        keychain_entry = {
            'serverName': data['server_name'],
            'serverUrl': data['server_url'],
            'clientId': data['client_id'],
            'accessToken': data['access_token'],
            'expiresAt': data['expires_at'],
            'refreshToken': data['refresh_token'],
            'scope': data['scope'],
        }
        result[kc_key] = keychain_entry
    return result


# =============================================================================
# Current Login Detection
# =============================================================================


def get_current_login_name() -> str | None:
    """Detect which login matches current disk state."""
    current = read_oauth_account()
    if not current:
        return None

    for login in list_all_logins():
        if (
            login.oauth_account.account_uuid == current.account_uuid
            and login.oauth_account.organization_uuid == current.organization_uuid
        ):
            return login.name

    return None


# =============================================================================
# Display Helpers
# =============================================================================


def _redact(val: str) -> str:
    """Redact token for display."""
    if len(val) > 16:
        return f'{val[:8]}...{val[-4:]}'
    return '***'


def _format_expiry(expires_ms: int | None) -> str:
    """Format expiry as human-readable time remaining."""
    if not expires_ms:
        return 'no expiry'
    expires_dt = datetime.fromtimestamp(expires_ms / 1000, UTC)
    remaining = expires_dt - datetime.now(UTC)
    if remaining.total_seconds() <= 0:
        return 'expired'
    days = remaining.days
    hours = remaining.seconds // 3600
    if days > 0:
        return f'{days}d {hours}h'
    if hours > 0:
        mins = (remaining.seconds % 3600) // 60
        return f'{hours}h {mins}m'
    return f'{remaining.seconds // 60}m'


def _format_plan(sub: str | None, tier: str | None) -> str:
    """Format subscription type and rate limit tier as friendly plan name."""
    if not sub:
        return 'Unknown'
    if not tier:
        return sub.title()
    multiplier = ''
    if tier.startswith('default_claude_max_'):
        multiplier = ' ' + tier.removeprefix('default_claude_max_')
    if sub == 'pro':
        return 'Pro'
    if sub == 'max':
        return f'Max{multiplier}'
    if sub == 'team':
        return f'Team{multiplier}'
    return f'{sub}{multiplier}'


def _format_setup_token_remaining(created_at: datetime | None) -> str:
    """Format setup-token remaining life from creation timestamp."""
    if not created_at:
        return 'unknown expiry'
    elapsed = datetime.now(UTC) - created_at
    remaining = SETUP_TOKEN_LIFETIME_DAYS - elapsed.days
    if remaining <= 0:
        return 'expired'
    return f'~{remaining}d remaining'


def _print_login(login: LoginFile, is_current: bool) -> None:
    """Print login summary."""
    marker = '*' if is_current else ' '
    parts: list[str] = [f'{marker} {login.name}']

    # Plan
    if login.claude_ai_oauth:
        parts.append(_format_plan(login.claude_ai_oauth.subscription_type, login.claude_ai_oauth.rate_limit_tier))
    elif login.api_key:
        parts.append('Console')

    # Auth methods (first listed = default for switch-login)
    if login.setup_token:
        remaining = _format_setup_token_remaining(login.setup_token_created_at)
        auth = f'setup-token ({remaining})'
        if login.claude_ai_oauth:
            auth += ' + keychain'
        parts.append(auth)
    elif login.claude_ai_oauth:
        expiry = _format_expiry(login.claude_ai_oauth.expires_at)
        has_refresh = bool(login.claude_ai_oauth.refresh_token)
        refresh_note = ', refresh ✓' if has_refresh else ', no refresh!'
        parts.append(f'keychain ({expiry}{refresh_note})')
    elif login.api_key:
        parts.append(f'api-key ({_redact(login.api_key)})')

    print(' | '.join(parts))


def _print_mcp_auth(name: str, entry: McpOAuthEntry) -> None:
    """Print MCP auth summary."""
    expiry = _format_expiry(entry.expires_at)
    has_refresh = bool(entry.refresh_token)
    refresh_note = ' (has refresh token)' if has_refresh else ' (no refresh token!)'
    print(f'  {name}')
    print(f'    {entry.server_url} | access expires: {expiry}{refresh_note}')


# =============================================================================
# Commands
# =============================================================================


def cmd_save_login(force: bool, inject_mcp: bool) -> None:
    """Save current Claude auth as a login file."""
    account = read_oauth_account()
    if not account:
        print('ERROR: No oauthAccount found. Run /login first.', file=sys.stderr)
        sys.exit(1)

    kc = read_keychain_raw()
    oauth_raw = kc.get('claudeAiOauth') if kc else None

    oauth = None
    if oauth_raw and oauth_raw.get('refreshToken'):
        # Only parse as ClaudeAiOAuth when refreshToken is present (real OAuth).
        # Setup-tokens in keychain have refreshToken: null — skip those.
        oauth = ClaudeAiOAuth.model_validate(
            {
                'access_token': oauth_raw['accessToken'],
                'refresh_token': oauth_raw['refreshToken'],
                'expires_at': oauth_raw['expiresAt'],
                'scopes': oauth_raw.get('scopes', []),
                'subscription_type': oauth_raw.get('subscriptionType'),
                'rate_limit_tier': oauth_raw.get('rateLimitTier'),
            }
        )

    # Capture API key for Console accounts
    api_key = read_api_key_keychain()
    if api_key and oauth:
        print('WARNING: Both OAuth tokens and API key found (auth conflict state).', file=sys.stderr)

    name = derive_login_name(account, oauth)

    path = login_path(name)
    existing_login = None
    if path.exists():
        existing_login = LoginFile.model_validate_json(path.read_text())
        if not force and existing_login.claude_ai_oauth and oauth:
            if existing_login.claude_ai_oauth.refresh_token != oauth.refresh_token:
                print(f'ERROR: Login "{name}" exists with different tokens. Use --force to overwrite.', file=sys.stderr)
                sys.exit(1)

    # Preserve setup_token from existing login
    setup_token = existing_login.setup_token if existing_login else None
    setup_token_created_at = existing_login.setup_token_created_at if existing_login else None

    login = LoginFile(
        name=name,
        created_at=datetime.now(UTC),
        oauth_account=account,
        claude_ai_oauth=oauth,
        api_key=api_key,
        setup_token=setup_token,
        setup_token_created_at=setup_token_created_at,
    )
    save_login(login)

    action = 'Updated' if existing_login else 'Created'
    print(f'{action} login: {name}')
    print(f'Path: {path}')

    if inject_mcp:
        mcp_auths = load_mcp_auths()
        if mcp_auths:
            kc_data = dict(kc) if kc else {}
            kc_data['mcpOAuth'] = dict(mcp_auths_to_keychain(mcp_auths))
            write_keychain_raw(kc_data)
            print(f'{len(mcp_auths)} MCP login(s) injected into keychain.')
        else:
            print('No saved MCP logins to inject.')


def cmd_switch_login(name: str, use_keychain: bool, restart: bool) -> None:
    """Switch to saved login. Enforces mutual exclusivity between auth types."""
    login = load_login(name)
    is_console = login.oauth_account.billing_type == 'prepaid'

    # Validate before any mutations — early exit must not leave partial state
    if is_console and use_keychain:
        print('ERROR: Console logins use API keys, not keychain OAuth.', file=sys.stderr)
        sys.exit(1)
    if not is_console and not login.setup_token and not login.claude_ai_oauth:
        print('ERROR: Login has no auth credentials.', file=sys.stderr)
        sys.exit(1)
    if not is_console and use_keychain and not login.claude_ai_oauth:
        print('ERROR: No keychain OAuth saved. Run /login and save-login first.', file=sys.stderr)
        sys.exit(1)

    # Capture previous login before overwriting
    previous_login = get_current_login_name()

    # Write oauthAccount to ~/.claude.json
    write_oauth_account(login.oauth_account)

    # Clean up primaryApiKey fallback regardless of account type
    remove_primary_api_key()

    # Load MCP auths for injection into credentials keychain
    mcp_auths = load_mcp_auths()

    auth_method = ''

    if is_console:
        if login.api_key:
            write_api_key_keychain(login.api_key)
        else:
            print('WARNING: No API key saved for this Console login.', file=sys.stderr)
            print('  You may need to run /login manually, then save-login --force.', file=sys.stderr)

        # Rewrite credentials with only MCP tokens (prefer fresh keychain data)
        current_kc = read_keychain_raw()
        current_mcp = current_kc.get('mcpOAuth') if current_kc else None
        if not current_mcp and mcp_auths:
            current_mcp = dict(mcp_auths_to_keychain(mcp_auths))
        if current_mcp:
            write_keychain_raw({'mcpOAuth': current_mcp})
        else:
            delete_keychain()

        auth_method = 'api-key'

    elif login.setup_token and not use_keychain:
        # Subscription with setup-token: write to keychain as claudeAiOauth.
        # The credential reader (dB) handles refreshToken:null gracefully —
        # the refresh mechanism silently skips when there's no refresh token.
        # This avoids the OAuth refresh race condition entirely.
        delete_api_key_keychain()

        kc_data: dict[str, Any] = {}
        kc_data['claudeAiOauth'] = {
            'accessToken': login.setup_token,
            'refreshToken': None,
            'expiresAt': None,
            'scopes': ['user:inference'],
            'subscriptionType': None,
            'rateLimitTier': None,
        }
        if mcp_auths:
            kc_data['mcpOAuth'] = dict(mcp_auths_to_keychain(mcp_auths))
        write_keychain_raw(kc_data)

        auth_method = 'setup-token'

    else:
        # Subscription with keychain OAuth (or --keychain forced)
        # Validated above — claude_ai_oauth is guaranteed non-None here
        oauth = login.claude_ai_oauth
        if not oauth:
            raise RuntimeError('unreachable: claude_ai_oauth validated at function entry')
        delete_api_key_keychain()

        kc_data = {}
        kc_data['claudeAiOauth'] = {
            'accessToken': oauth.access_token,
            'refreshToken': oauth.refresh_token,
            'expiresAt': oauth.expires_at,
            'scopes': list(oauth.scopes),
            'subscriptionType': oauth.subscription_type,
            'rateLimitTier': oauth.rate_limit_tier,
        }
        if mcp_auths:
            kc_data['mcpOAuth'] = dict(mcp_auths_to_keychain(mcp_auths))
        write_keychain_raw(kc_data)

        auth_method = 'keychain'

    # Write switch-pending marker for statusline to detect
    marker: dict[str, Any] = {
        'emailAddress': login.oauth_account.email_address,
        'authMethod': auth_method,
        'loginName': login.name,
    }
    if login.oauth_account.billing_type:
        marker['billingType'] = login.oauth_account.billing_type
    if login.claude_ai_oauth:
        # Title-case to match statusline's _read_static_data display format
        sub = login.claude_ai_oauth.subscription_type or ''
        marker['subscription'] = {
            'free': 'Free',
            'pro': 'Pro',
            'team': 'Team',
            'max': 'Max',
            'enterprise': 'Enterprise',
        }.get(sub, sub)
        tier = login.claude_ai_oauth.rate_limit_tier
        if tier:
            marker['tier'] = tier
    if previous_login:
        marker['previousLogin'] = previous_login
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_file_atomically(SWITCH_PENDING_PATH, json.dumps(marker))

    print(f'Switched to: {login.name}')
    print(f'  {login.oauth_account.email_address}')
    print(f'  Auth: {auth_method}')
    if is_console:
        key_status = 'injected' if login.api_key else 'MISSING'
        print(f'  Console (prepaid) | API key: {key_status}')
    elif login.claude_ai_oauth:
        print(f'  {login.claude_ai_oauth.subscription_type} ({login.claude_ai_oauth.rate_limit_tier})')
    if mcp_auths:
        print(f'  {len(mcp_auths)} MCP auth(s) injected')
    print()

    if restart:
        try:
            claude_pid = find_claude_pid()
            cwd = os.getcwd()
            session_id = resolve_session_id(claude_pid, cwd)
            _spawn_kill_and_copy_resume(claude_pid, session_id)
            print(f'Killing Claude Code (PID {claude_pid})...')
            print(f'Resume command copied to clipboard: claude --resume {session_id}')
            print('Paste (Cmd+V) + Enter after Claude exits.')
        except RuntimeError as e:
            print(f'WARNING: Cannot auto-launch: {e}', file=sys.stderr)
            print('Restart Claude Code manually to activate.')
    else:
        print('Restart Claude Code to activate.')


def cmd_list_logins() -> None:
    """List all saved logins."""
    logins = list_all_logins()
    if not logins:
        print('No saved logins.')
        return

    current = get_current_login_name()
    print(f'{len(logins)} saved login(s):')
    print()
    for login in logins:
        _print_login(login, login.name == current)


def cmd_current_login() -> None:
    """Show current login from disk."""
    account = read_oauth_account()
    api_key = read_api_key_keychain()

    if not account:
        print('Not logged in (no oauthAccount).')
        if api_key:
            print(f'  WARNING: Stale API key in keychain: {_redact(api_key)}')
        return

    kc = read_keychain_raw()
    oauth_raw = kc.get('claudeAiOauth') if kc else None

    # Check switch-pending state
    pending_marker = None
    if SWITCH_PENDING_PATH.exists():
        pending_marker = json.loads(SWITCH_PENDING_PATH.read_text())

    if pending_marker:
        previous = pending_marker.get('previousLogin')
        if previous:
            print(f'Active login: {previous}')
        print()
        print('Pending login (restart to activate):')
    else:
        print('Current login (from disk):')

    print(f'  {account.email_address}')
    print(f'  {account.billing_type}')

    if oauth_raw:
        access = oauth_raw.get('accessToken', '')
        has_refresh = bool(oauth_raw.get('refreshToken'))
        sub = oauth_raw.get('subscriptionType', '?')
        tier = oauth_raw.get('rateLimitTier', '?')
        if not has_refresh:
            # Setup-tokens are written with refreshToken: null
            print(f'  Setup token: {_redact(access)} (in keychain)')
        else:
            print(f'  OAuth: {sub} ({tier})')

    if api_key:
        print(f'  API key: {_redact(api_key)}')

    # Detect auth conflicts (API key + OAuth in credentials keychain)
    if oauth_raw and api_key:
        print()
        print('  WARNING: Auth conflict! Both OAuth and API key active.')
        print('  Use switch-login to cleanly switch accounts.')

    # Check if it matches a saved login
    current_name = get_current_login_name()
    if current_name:
        print(f'  Matches: {current_name}')
    else:
        print('  (not saved as a login)')


def cmd_delete_login(name: str) -> None:
    """Delete a saved login."""
    # Verify exists
    if not login_path(name).exists():
        print(f'Login not found: {name}', file=sys.stderr)
        sys.exit(1)

    delete_login_file(name)
    print(f'Deleted: {name}')


def cmd_save_mcp_login(server_query: str) -> None:
    """Save MCP server token from keychain."""
    kc = read_keychain_raw()
    if not kc:
        print('ERROR: No keychain entry found.', file=sys.stderr)
        sys.exit(1)

    mcp_oauth = kc.get('mcpOAuth')
    if not mcp_oauth:
        print('ERROR: No mcpOAuth in keychain. Auth the server via /mcp first.', file=sys.stderr)
        sys.exit(1)

    # Find matching entry
    matches = []
    for kc_key, entry in mcp_oauth.items():
        server_name = entry.get('serverName', '')
        if server_query.lower() in server_name.lower():
            matches.append((kc_key, entry))

    if not matches:
        print(f'No MCP server matching "{server_query}"', file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f'Ambiguous match for "{server_query}":', file=sys.stderr)
        for kc_key, entry in matches:
            print(f'  {entry.get("serverName", kc_key)}', file=sys.stderr)
        sys.exit(1)

    kc_key, entry_raw = matches[0]

    # Convert to McpOAuthEntry
    entry = McpOAuthEntry.model_validate(
        {
            'server_name': entry_raw['serverName'],
            'server_url': entry_raw['serverUrl'],
            'client_id': entry_raw['clientId'],
            'access_token': entry_raw['accessToken'],
            'expires_at': entry_raw['expiresAt'],
            'refresh_token': entry_raw['refreshToken'],
            'scope': entry_raw['scope'],
            'keychain_key': kc_key,
        }
    )

    # Load, update, save
    auths = dict(load_mcp_auths())
    auths[entry.server_name] = entry
    save_mcp_auths(auths)

    print(f'Saved MCP login: {entry.server_name}')
    print(f'  Expires: {_format_expiry(entry.expires_at)}')


def cmd_list_mcp_logins() -> None:
    """List all saved MCP logins."""
    auths = load_mcp_auths()
    if not auths:
        print('No saved MCP logins.')
        return

    print(f'{len(auths)} saved MCP login(s):')
    print()
    for name, entry in auths.items():
        _print_mcp_auth(name, entry)


def cmd_delete_mcp_login(server_query: str) -> None:
    """Delete saved MCP login."""
    auths = dict(load_mcp_auths())
    if not auths:
        print('No saved MCP logins.', file=sys.stderr)
        sys.exit(1)

    # Match by name
    matches = [name for name in auths if server_query.lower() in name.lower()]
    if not matches:
        print(f'No MCP auth matching "{server_query}"', file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f'Ambiguous match for "{server_query}":', file=sys.stderr)
        for name in matches:
            print(f'  {name}', file=sys.stderr)
        sys.exit(1)

    name = matches[0]
    del auths[name]
    save_mcp_auths(auths)

    print(f'Deleted MCP login: {name}')


def cmd_save_setup_token(token: str, login_id: str) -> None:
    """Save a long-lived setup token to an existing login."""
    if not token.startswith(SETUP_TOKEN_PREFIX):
        print(f'ERROR: Invalid token format. Must start with {SETUP_TOKEN_PREFIX}', file=sys.stderr)
        sys.exit(1)

    login = load_login(login_id)

    if login.oauth_account.billing_type == 'prepaid':
        print('ERROR: Setup tokens are only for subscription accounts (Pro/Max/Team).', file=sys.stderr)
        sys.exit(1)

    updated = login.model_copy(
        update={
            'setup_token': token,
            'setup_token_created_at': datetime.now(UTC),
        },
    )
    save_login(updated)

    print(f'Saved setup-token to: {login_id}')
    print(f'  {_format_setup_token_remaining(updated.setup_token_created_at)}')


def cmd_clear_setup_token(login_id: str) -> None:
    """Remove setup token from a login."""
    login = load_login(login_id)

    if not login.setup_token:
        print(f'ERROR: No setup token on this login: {login_id}', file=sys.stderr)
        sys.exit(1)

    updated = login.model_copy(update={'setup_token': None, 'setup_token_created_at': None})
    save_login(updated)

    print(f'Removed setup-token from: {login_id}')
    if updated.claude_ai_oauth:
        print('  switch-login will now use keychain OAuth.')
    elif not updated.api_key:
        print('  WARNING: Login has no auth credentials.', file=sys.stderr)


def cmd_nuke_claude_auth() -> None:
    """Wipe all Claude Code auth state. Does not affect saved logins."""
    delete_keychain()
    delete_api_key_keychain()
    remove_oauth_account()
    remove_primary_api_key()
    print('Claude Code auth state nuked.')
    print('  Keychain "Claude Code-credentials": cleared')
    print('  Keychain "Claude Code" (API key):   cleared')
    print('  oauthAccount: removed')
    print('  primaryApiKey: removed')
    print()
    print('Saved logins and MCP logins are untouched.')
    print('Restart Claude Code to take effect.')


# =============================================================================
# Tab Completion
# =============================================================================


def _complete_login_id(incomplete: str) -> Sequence[tuple[str, str]]:
    """Complete login IDs from saved logins. Returns (name, plan) tuples."""
    completions: list[tuple[str, str]] = []
    for login in list_all_logins():
        if not login.name.lower().startswith(incomplete.lower()):
            continue
        if login.claude_ai_oauth:
            help_text = _format_plan(login.claude_ai_oauth.subscription_type, login.claude_ai_oauth.rate_limit_tier)
        else:
            help_text = 'Console'
        completions.append((login.name, help_text))
    return completions


def _complete_keychain_mcp_server(incomplete: str) -> Sequence[tuple[str, str]]:
    """Complete MCP server names from current keychain (for save-mcp-login)."""
    kc = read_keychain_raw()
    if not kc:
        return []
    mcp_oauth = kc.get('mcpOAuth')
    if not mcp_oauth or not isinstance(mcp_oauth, dict):
        return []
    completions: list[tuple[str, str]] = []
    for entry in mcp_oauth.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get('serverName', '')
        if name and name.lower().startswith(incomplete.lower()):
            url = entry.get('serverUrl', '')
            completions.append((name, url))
    return completions


def _complete_saved_mcp_server(incomplete: str) -> Sequence[tuple[str, str]]:
    """Complete MCP server names from saved MCP auths (for delete-mcp-login)."""
    auths = load_mcp_auths()
    return [(name, entry.server_url) for name, entry in auths.items() if name.lower().startswith(incomplete.lower())]


def _complete_subscription_login_id(incomplete: str) -> Sequence[tuple[str, str]]:
    """Subscription logins only (for save-setup-token --login)."""
    completions: list[tuple[str, str]] = []
    for login in list_all_logins():
        if login.oauth_account.billing_type == 'prepaid':
            continue
        if not login.name.lower().startswith(incomplete.lower()):
            continue
        if login.claude_ai_oauth:
            help_text = _format_plan(login.claude_ai_oauth.subscription_type, login.claude_ai_oauth.rate_limit_tier)
        else:
            help_text = 'Subscription'
        completions.append((login.name, help_text))
    return completions


def _complete_setup_token_login_id(incomplete: str) -> Sequence[tuple[str, str]]:
    """Logins that have setup tokens (for clear-setup-token)."""
    completions: list[tuple[str, str]] = []
    for login in list_all_logins():
        if not login.setup_token:
            continue
        if not login.name.lower().startswith(incomplete.lower()):
            continue
        remaining = _format_setup_token_remaining(login.setup_token_created_at)
        completions.append((login.name, remaining))
    return completions


# =============================================================================
# CLI
# =============================================================================

# Typer's built-in --install-completion/--show-completion are disabled:
# - --show-completion uses shellingham process-tree detection (fails outside interactive
#   shells, including Claude Code's Bash tool, CI runners, and editor terminals)
# - --install-completion hardcodes ~/.zfunc/~/.zshrc (ignores ZDOTDIR)
# The `completion` command replaces both with explicit shell argument and ZDOTDIR support.
app = typer.Typer(help='Claude Code login and MCP auth manager.', add_completion=False)
typer.completion.completion_init()  # Patch click for runtime tab completion (skipped by add_completion=False)


@app.callback(invoke_without_command=True)
def _app_main(ctx: typer.Context) -> None:
    """Run migrations before any command."""
    migrate_login_files()
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise SystemExit(0)


# -- Saved login management --------------------------------------------------


@app.command('save-login')
def cli_save_login(
    force: bool = typer.Option(False, '--force', help='Overwrite if tokens differ'),
    inject_mcp: bool = typer.Option(False, '--inject-mcp', help='Re-inject saved MCP tokens into keychain'),
) -> None:
    """Save current Claude auth as a login file."""
    cmd_save_login(force, inject_mcp)


@app.command('switch-login')
def cli_switch_login(
    login_id: str | None = typer.Argument(None, help='Login ID', autocompletion=_complete_login_id),
    use_keychain: bool = typer.Option(False, '--keychain', help='Force keychain OAuth (bypass setup-token)'),
    restart: bool = typer.Option(False, '--restart', help='Kill Claude Code and copy resume command to clipboard'),
) -> None:
    """Switch to saved login + inject MCP tokens."""
    if login_id is None:
        logins = list_all_logins()
        current = get_current_login_name()
        console = rich.console.Console(stderr=True)
        msg = click.MissingParameter(param_type='argument', param_hint="'LOGIN_ID'").format_message()
        if logins:
            lines = [f'{msg}\n\nAvailable logins:']
            for login in sorted(logins, key=lambda x: x.name):
                marker = '*' if login.name == current else ' '
                if login.claude_ai_oauth:
                    plan = _format_plan(login.claude_ai_oauth.subscription_type, login.claude_ai_oauth.rate_limit_tier)
                    lines.append(f'{marker} {login.name} | {plan}')
                else:
                    lines.append(f'{marker} {login.name}')
            console.print(rich.panel.Panel('\n'.join(lines), border_style='red', title='Error', title_align='left'))
        else:
            console.print(
                rich.panel.Panel(f'{msg} No saved logins.', border_style='red', title='Error', title_align='left')
            )
        raise SystemExit(1)
    cmd_switch_login(login_id, use_keychain, restart)


@app.command('list-logins')
def cli_list_logins() -> None:
    """List all saved logins."""
    cmd_list_logins()


@app.command('delete-login')
def cli_delete_login(login_id: str = typer.Argument(..., help='Login ID', autocompletion=_complete_login_id)) -> None:
    """Delete a saved login."""
    cmd_delete_login(login_id)


@app.command('save-setup-token')
def cli_save_setup_token(
    login_id: str = typer.Option(..., '--login', help='Login ID', autocompletion=_complete_subscription_login_id),
    token: str = typer.Option('', '--token', help='Token (prompted securely if omitted)'),
) -> None:
    """Save long-lived token to a login (subscription accounts only).

    Token is prompted interactively to avoid shell history exposure.
    Or pass via --token or stdin: echo TOKEN | claude-login save-setup-token --login ID --token -
    """
    if not token:
        if not sys.stdin.isatty():
            token = sys.stdin.read().strip()
        else:
            token = typer.prompt('Paste setup token', hide_input=True)
    elif token == '-':
        token = sys.stdin.read().strip()
    if not token:
        print('ERROR: No token provided.', file=sys.stderr)
        sys.exit(1)
    cmd_save_setup_token(token, login_id)


@app.command('clear-setup-token')
def cli_clear_setup_token(
    login_id: str = typer.Argument(..., help='Login ID', autocompletion=_complete_setup_token_login_id),
) -> None:
    """Remove setup-token from a login."""
    cmd_clear_setup_token(login_id)


# -- Saved MCP login management ----------------------------------------------


@app.command('save-mcp-login')
def cli_save_mcp_login(
    server: str = typer.Argument(..., help='MCP server name', autocompletion=_complete_keychain_mcp_server),
) -> None:
    """Save MCP server token from keychain."""
    cmd_save_mcp_login(server)


@app.command('list-mcp-logins')
def cli_list_mcp_logins() -> None:
    """List saved MCP logins."""
    cmd_list_mcp_logins()


@app.command('delete-mcp-login')
def cli_delete_mcp_login(
    server: str = typer.Argument(..., help='MCP server name', autocompletion=_complete_saved_mcp_server),
) -> None:
    """Remove saved MCP login."""
    cmd_delete_mcp_login(server)


# -- Claude Code auth state ---------------------------------------------------


@app.command('current-claude-login')
def cli_current_claude_login() -> None:
    """Show Claude Code's active auth from disk."""
    cmd_current_login()


@app.command('nuke-claude-auth')
def cli_nuke_claude_auth() -> None:
    """Wipe Claude Code's active auth state. Saved logins are untouched."""
    cmd_nuke_claude_auth()


# -- Shell completion ---------------------------------------------------------

# Custom install-path overrides. Shells listed here bypass typer's built-in
# install(), which handles path resolution but may hardcode paths or mutate
# rc files. Add entries here to fix broken defaults for specific shells.
# Unlisted shells delegate to typer.completion.install() as-is.
COMPLETION_INSTALL_OVERRIDES: Mapping[str, Callable[[str], Path]] = {
    # typer hardcodes ~/.zfunc and appends to ~/.zshrc — override to respect ZDOTDIR.
    'zsh': lambda prog: Path(os.environ.get('ZDOTDIR') or Path.home() / '.zsh') / 'completions' / f'_{prog}',
    # typer appends a source line to ~/.bashrc — use XDG path that bash-completion auto-loads.
    'bash': lambda prog: Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
    / 'bash-completion'
    / 'completions'
    / prog,
}


@app.command('completion')
def cli_completion(
    shell: typer.completion.Shells = typer.Argument(..., help='Shell type'),
    install: bool = typer.Option(False, '--install', help='Write to shell-standard location'),
) -> None:
    """Print or install shell tab completions.

    Prints the completion script to stdout. With --install, writes to the
    standard completions directory for the specified shell.

    Zsh: $ZDOTDIR/completions/ (falls back to ~/.zsh/completions/)
    """
    ctx = click.get_current_context()
    prog_name = ctx.find_root().info_name or 'claude-login'
    complete_var = f'_{prog_name.replace("-", "_").upper()}_COMPLETE'
    script = typer.completion.get_completion_script(
        prog_name=prog_name,
        complete_var=complete_var,
        shell=shell.value,
    )
    if not install:
        click.echo(script)
        return
    resolver = COMPLETION_INSTALL_OVERRIDES.get(shell.value)
    if resolver is not None:
        dest = resolver(prog_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(script)
    else:
        _, dest = typer.completion.install(
            shell=shell.value,
            prog_name=prog_name,
            complete_var=complete_var,
        )
    rich.console.Console(stderr=True).print(f'Installed to [bold]{dest}[/bold]\nRestart shell to activate.')


if __name__ == '__main__':
    # Derive clean command name for consistent help text and tab completion,
    # regardless of invocation method (direct .py, symlink, or launcher).
    app(prog_name=os.path.basename(sys.argv[0]).removesuffix('.py'))
