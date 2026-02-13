#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["pydantic>=2.0", "typer>=0.9.0"]
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

Login IDs are auto-derived: email--{Console,Team,Personal}

Run directly from GitHub (no clone needed):
    uv run https://raw.githubusercontent.com/chrisguillory/claude-workspace/main/scripts/claude-login.py list-logins

Pin to a specific commit for deterministic behavior:
    uv run https://raw.githubusercontent.com/chrisguillory/claude-workspace/<sha>/scripts/claude-login.py list-logins

Possible improvements:
- Keychain `-T` flag: pass `-T /path/to/claude` to explicitly grant access.
  Not currently needed (no ACL restrictions observed). See:
  https://macromates.com/blog/2006/keychain-access-from-shell/
- Tab completion for switch-login LOGIN_ID via typer autocompletion callback.
- Cross-platform: Linux uses ~/.claude/.credentials.json instead of Keychain.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import pydantic
import rich.console
import rich.panel
import typer

# =============================================================================
# Pydantic Models
# =============================================================================


class StrictModel(pydantic.BaseModel):
    """For data we own — fail on unknown fields."""

    model_config = pydantic.ConfigDict(extra='forbid', strict=True, frozen=True)


class PermissiveModel(pydantic.BaseModel):
    """For Claude Code data — allow unknown fields for forward compatibility."""

    model_config = pydantic.ConfigDict(extra='allow', frozen=True)


class OAuthAccount(PermissiveModel):
    """Claude Code oauthAccount from ~/.claude.json."""

    # Always present
    account_uuid: str
    email_address: str
    organization_uuid: str
    billing_type: str
    has_extra_usage_enabled: bool
    display_name: str
    subscription_created_at: str

    # Sometimes present
    organization_name: str | None = None
    organization_role: str | None = None
    workspace_role: str | None = None
    account_created_at: str | None = None


class ClaudeAiOAuth(PermissiveModel):
    """Claude account OAuth tokens from keychain."""

    access_token: str
    refresh_token: str
    expires_at: int
    scopes: Sequence[str]
    subscription_type: str
    rate_limit_tier: str


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
    """Saved login credentials."""

    name: str
    created_at: datetime
    oauth_account: OAuthAccount
    claude_ai_oauth: ClaudeAiOAuth | None = None
    api_key: str | None = None


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
    """Delete keychain entry entirely."""
    subprocess.run(
        ['security', 'delete-generic-password', '-s', KEYCHAIN_SERVICE_CREDENTIALS],
        capture_output=True,
        timeout=5,
    )


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


def read_oauth_account() -> OAuthAccount | None:
    """Read oauthAccount from ~/.claude.json."""
    if not CONFIG_PATH.is_file():
        return None
    config = json.loads(CONFIG_PATH.read_text())
    raw = config.get('oauthAccount')
    if not raw:
        return None
    # Convert camelCase keys to snake_case for Pydantic
    converted = {
        'account_uuid': raw.get('accountUuid'),
        'email_address': raw.get('emailAddress'),
        'organization_uuid': raw.get('organizationUuid'),
        'billing_type': raw.get('billingType'),
        'has_extra_usage_enabled': raw.get('hasExtraUsageEnabled'),
        'display_name': raw.get('displayName'),
        'subscription_created_at': raw.get('subscriptionCreatedAt'),
        'organization_name': raw.get('organizationName'),
        'organization_role': raw.get('organizationRole'),
        'workspace_role': raw.get('workspaceRole'),
        'account_created_at': raw.get('accountCreatedAt'),
    }
    # Remove None values for optional fields
    converted = {k: v for k, v in converted.items() if v is not None}
    return OAuthAccount.model_validate(converted)


def write_oauth_account(account: OAuthAccount) -> None:
    """Write oauthAccount to ~/.claude.json, preserving all other fields."""
    config = json.loads(CONFIG_PATH.read_text())
    # Convert snake_case back to camelCase for Claude Code
    dumped = account.model_dump()
    camel = {
        'accountUuid': dumped['account_uuid'],
        'emailAddress': dumped['email_address'],
        'organizationUuid': dumped['organization_uuid'],
        'billingType': dumped['billing_type'],
        'hasExtraUsageEnabled': dumped['has_extra_usage_enabled'],
        'displayName': dumped['display_name'],
        'subscriptionCreatedAt': dumped['subscription_created_at'],
    }
    # Add optional fields if present
    for snake, camel_key in [
        ('organization_name', 'organizationName'),
        ('organization_role', 'organizationRole'),
        ('workspace_role', 'workspaceRole'),
        ('account_created_at', 'accountCreatedAt'),
    ]:
        if dumped.get(snake) is not None:
            camel[camel_key] = dumped[snake]

    config['oauthAccount'] = camel
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def remove_oauth_account() -> None:
    """Remove oauthAccount from ~/.claude.json."""
    config = json.loads(CONFIG_PATH.read_text())
    config.pop('oauthAccount', None)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def remove_primary_api_key() -> None:
    """Remove primaryApiKey from ~/.claude.json if present.

    Claude Code stores this as a fallback when keychain write fails.
    Must be cleaned up when switching to subscription accounts.
    """
    config = json.loads(CONFIG_PATH.read_text())
    if 'primaryApiKey' in config:
        del config['primaryApiKey']
        CONFIG_PATH.write_text(json.dumps(config, indent=2))


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
    path.write_text(login.model_dump_json(indent=2))
    path.chmod(0o600)
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
    # Serialize with model_dump
    serialized = {k: v.model_dump() for k, v in auths.items()}
    MCP_AUTHS_PATH.write_text(json.dumps(serialized, indent=2))
    MCP_AUTHS_PATH.chmod(0o600)


def load_mcp_auths() -> dict[str, McpOAuthEntry]:  # strict_typing_linter.py: mutable-type
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


def _format_expiry(expires_ms: int) -> str:
    """Format expiry as human-readable time remaining."""
    if expires_ms == 0:
        return 'no expiry'
    expires_dt = datetime.fromtimestamp(expires_ms / 1000, UTC)
    remaining = expires_dt - datetime.now(UTC)
    days = remaining.days
    hours = remaining.seconds // 3600
    if days > 0:
        return f'{days}d {hours}h'
    if hours > 0:
        mins = (remaining.seconds % 3600) // 60
        return f'{hours}h {mins}m'
    return f'{remaining.seconds // 60}m'


def _format_plan(sub: str, tier: str) -> str:
    """Format subscription type and rate limit tier as friendly plan name."""
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


def _print_login(login: LoginFile, is_current: bool) -> None:
    """Print login summary."""
    marker = '*' if is_current else ' '

    if login.claude_ai_oauth:
        plan = _format_plan(login.claude_ai_oauth.subscription_type, login.claude_ai_oauth.rate_limit_tier)
        expiry = _format_expiry(login.claude_ai_oauth.expires_at)
        has_refresh = bool(login.claude_ai_oauth.refresh_token)
        refresh_note = ' (has refresh token)' if has_refresh else ' (no refresh token!)'
        print(f'{marker} {login.name} | {plan} | access expires: {expiry}{refresh_note}')
    else:
        print(f'{marker} {login.name}')


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
    if oauth_raw:
        oauth = ClaudeAiOAuth.model_validate(
            {
                'access_token': oauth_raw['accessToken'],
                'refresh_token': oauth_raw['refreshToken'],
                'expires_at': oauth_raw['expiresAt'],
                'scopes': oauth_raw['scopes'],
                'subscription_type': oauth_raw['subscriptionType'],
                'rate_limit_tier': oauth_raw['rateLimitTier'],
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

    login = LoginFile(
        name=name,
        created_at=datetime.now(UTC),
        oauth_account=account,
        claude_ai_oauth=oauth,
        api_key=api_key,
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


def cmd_switch_login(name: str) -> None:
    """Switch to saved login. Enforces mutual exclusivity between auth types."""
    login = load_login(name)
    is_console = login.oauth_account.billing_type == 'prepaid'

    # Capture previous login before overwriting
    previous_login = get_current_login_name()

    # Write oauthAccount to ~/.claude.json
    write_oauth_account(login.oauth_account)

    # Clean up primaryApiKey fallback regardless of account type
    remove_primary_api_key()

    # Load MCP auths for injection into credentials keychain
    mcp_auths = load_mcp_auths()

    if is_console:
        # Console: write API key, strip OAuth from credentials
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
    else:
        # Subscription: write OAuth, delete API key
        delete_api_key_keychain()

        kc_data = {}
        if login.claude_ai_oauth:
            kc_data['claudeAiOauth'] = {
                'accessToken': login.claude_ai_oauth.access_token,
                'refreshToken': login.claude_ai_oauth.refresh_token,
                'expiresAt': login.claude_ai_oauth.expires_at,
                'scopes': list(login.claude_ai_oauth.scopes),
                'subscriptionType': login.claude_ai_oauth.subscription_type,
                'rateLimitTier': login.claude_ai_oauth.rate_limit_tier,
            }
        if mcp_auths:
            kc_data['mcpOAuth'] = dict(mcp_auths_to_keychain(mcp_auths))
        if kc_data:
            write_keychain_raw(kc_data)
        else:
            delete_keychain()

    # Write switch-pending marker for statusline to detect
    marker: dict[str, str] = {
        'emailAddress': login.oauth_account.email_address,
        'billingType': login.oauth_account.billing_type,
    }
    if login.claude_ai_oauth:
        # Title-case to match statusline's _read_static_data display format
        sub = login.claude_ai_oauth.subscription_type
        marker['subscription'] = {
            'free': 'Free',
            'pro': 'Pro',
            'team': 'Team',
            'max': 'Max',
            'enterprise': 'Enterprise',
        }.get(sub, sub)
        marker['tier'] = login.claude_ai_oauth.rate_limit_tier
    if previous_login:
        marker['previousLogin'] = previous_login
    SWITCH_PENDING_PATH.write_text(json.dumps(marker))

    print(f'Switched to: {login.name}')
    print(f'  {login.oauth_account.email_address}')
    if is_console:
        key_status = 'injected' if login.api_key else 'MISSING'
        print(f'  Console (prepaid) | API key: {key_status}')
    elif login.claude_ai_oauth:
        print(f'  {login.claude_ai_oauth.subscription_type} ({login.claude_ai_oauth.rate_limit_tier})')
    if mcp_auths:
        print(f'  {len(mcp_auths)} MCP auth(s) injected')
    print()
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
        sub = oauth_raw.get('subscriptionType', '?')
        tier = oauth_raw.get('rateLimitTier', '?')
        print(f'  OAuth: {sub} ({tier})')

    if api_key:
        print(f'  API key: {_redact(api_key)}')

    if oauth_raw and api_key:
        print()
        print('  WARNING: Auth conflict! Both OAuth and API key are present.')
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
    auths = load_mcp_auths()
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
    auths = load_mcp_auths()
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
# CLI
# =============================================================================

app = typer.Typer(help='Claude Code login and MCP auth manager.')


# -- Saved login management --------------------------------------------------


@app.command('save-login')
def cli_save_login(
    force: bool = typer.Option(False, '--force', help='Overwrite if tokens differ'),
    inject_mcp: bool = typer.Option(False, '--inject-mcp', help='Re-inject saved MCP tokens into keychain'),
) -> None:
    """Save current Claude auth as a login file."""
    cmd_save_login(force, inject_mcp)


@app.command('switch-login')
def cli_switch_login(login_id: str | None = typer.Argument(None, help='Login ID')) -> None:
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
    cmd_switch_login(login_id)


@app.command('list-logins')
def cli_list_logins() -> None:
    """List all saved logins."""
    cmd_list_logins()


@app.command('delete-login')
def cli_delete_login(login_id: str = typer.Argument(..., help='Login ID')) -> None:
    """Delete a saved login."""
    cmd_delete_login(login_id)


# -- Saved MCP login management ----------------------------------------------


@app.command('save-mcp-login')
def cli_save_mcp_login(server: str = typer.Argument(..., help='MCP server name')) -> None:
    """Save MCP server token from keychain."""
    cmd_save_mcp_login(server)


@app.command('list-mcp-logins')
def cli_list_mcp_logins() -> None:
    """List saved MCP logins."""
    cmd_list_mcp_logins()


@app.command('delete-mcp-login')
def cli_delete_mcp_login(server: str = typer.Argument(..., help='MCP server name')) -> None:
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


if __name__ == '__main__':
    app()
