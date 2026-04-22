"""Claude CLI completion data — derived from binary analysis of v2.1.114.

This is a committed artifact, not a runtime generator. When Claude Code
updates, re-run the binary analysis workflow (documented in
``mcp/claude-session/CLAUDE.md`` § Unpacking the JS Bundle) and update
this file.

Analysis method:
    1. ``claude --help`` for documented flags
    2. ``npx tweakcc unpack`` (on a COPY of the binary) for full source
    3. Walk-back from every ``.hideHelp()`` call to the nearest flag
       name to enumerate hidden flags
    4. Subcommand ``--help`` for nested structures

Last fully verified: v2.1.114, 2026-04-18.
Spot-added post-v2.1.114: ``--allowedTools`` / ``--disallowedTools``
camelCase aliases (verified against v2.1.116 ``--help`` output).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Literal

__all__ = [
    'MODEL_ALIASES',
    'ROOT_FLAGS',
    'SUBCOMMANDS',
]

type ArgType = Literal['none', 'string', 'path', 'choice', 'session', 'model', 'json']


@dataclasses.dataclass(frozen=True, slots=True)
class FlagDef:
    """A CLI flag definition derived from binary analysis."""

    name: str
    short: str | None = None
    arg_type: ArgType = 'none'
    choices: Sequence[str] = ()
    documented: bool = True
    description: str = ''


@dataclasses.dataclass(frozen=True, slots=True)
class SubcommandDef:
    """A CLI subcommand definition."""

    name: str
    description: str = ''


# -- Root command flags --------------------------------------------------------

ROOT_FLAGS: Sequence[FlagDef] = [
    # Documented flags (from --help)
    FlagDef('--add-dir', arg_type='path', description='Additional directories to allow tool access to'),
    FlagDef('--agent', arg_type='string', description='Agent for the current session'),
    FlagDef('--agents', arg_type='json', description='JSON object defining custom agents'),
    FlagDef('--allow-dangerously-skip-permissions', description='Enable bypassing permission checks as an option'),
    FlagDef('--allowed-tools', arg_type='string', description='Comma or space-separated list of tool names to allow'),
    FlagDef('--allowedTools', arg_type='string', description='camelCase alias of --allowed-tools'),
    FlagDef('--append-system-prompt', arg_type='string', description='Append to the default system prompt'),
    FlagDef('--bare', description='Minimal mode: skip hooks, LSP, plugins, auto-memory'),
    FlagDef('--betas', arg_type='string', description='Beta headers for API requests'),
    FlagDef('--brief', description='Enable SendUserMessage tool for agent-to-user communication'),
    FlagDef('--chrome', description='Enable Claude in Chrome integration'),
    FlagDef('--continue', '-c', description='Continue most recent conversation'),
    FlagDef('--dangerously-skip-permissions', description='Bypass all permission checks'),
    FlagDef('--debug', '-d', arg_type='string', description='Enable debug mode with optional category filter'),
    FlagDef('--debug-file', arg_type='path', description='Write debug logs to specific file'),
    FlagDef('--disable-slash-commands', description='Disable all skills'),
    FlagDef('--disallowed-tools', arg_type='string', description='Comma or space-separated list of tool names to deny'),
    FlagDef('--disallowedTools', arg_type='string', description='camelCase alias of --disallowed-tools'),
    FlagDef(
        '--effort',
        arg_type='choice',
        choices=('low', 'medium', 'high', 'xhigh', 'max'),
        description='Effort level for the session',
    ),
    FlagDef(
        '--exclude-dynamic-system-prompt-sections',
        description='Move per-machine sections into first user message for prompt-cache reuse',
    ),
    FlagDef('--fallback-model', arg_type='model', description='Fallback model when default is overloaded'),
    FlagDef('--file', arg_type='string', description='File resources to download at startup'),
    FlagDef('--fork-session', description='Create new session ID when resuming'),
    FlagDef('--from-pr', arg_type='string', description='Resume session linked to a PR'),
    FlagDef('--help', '-h', description='Display help'),
    FlagDef('--ide', description='Auto-connect to IDE on startup'),
    FlagDef('--include-hook-events', description='Include hook lifecycle events in output stream'),
    FlagDef('--include-partial-messages', description='Include partial message chunks'),
    FlagDef(
        '--input-format',
        arg_type='choice',
        choices=('text', 'stream-json'),
        description='Input format for --print mode',
    ),
    FlagDef('--json-schema', arg_type='json', description='JSON Schema for structured output validation'),
    FlagDef('--max-budget-usd', arg_type='string', description='Maximum dollar amount for API calls'),
    FlagDef('--mcp-config', arg_type='path', description='Load MCP servers from JSON files'),
    FlagDef('--mcp-debug', description='(DEPRECATED) Enable MCP debug mode'),
    FlagDef('--model', arg_type='model', description='Model for the session'),
    FlagDef('--name', '-n', arg_type='string', description='Set display name for session'),
    FlagDef('--no-chrome', description='Disable Claude in Chrome'),
    FlagDef('--no-session-persistence', description='Disable session persistence'),
    FlagDef(
        '--output-format',
        arg_type='choice',
        choices=('text', 'json', 'stream-json'),
        description='Output format for --print mode',
    ),
    FlagDef(
        '--permission-mode',
        arg_type='choice',
        choices=('acceptEdits', 'auto', 'bypassPermissions', 'default', 'dontAsk', 'plan'),
        description='Permission mode',
    ),
    FlagDef('--plugin-dir', arg_type='path', description='Load plugins from directory'),
    FlagDef('--print', '-p', description='Print response and exit'),
    FlagDef(
        '--remote-control-session-name-prefix', arg_type='string', description='Prefix for Remote Control session names'
    ),
    FlagDef('--replay-user-messages', description='Re-emit user messages from stdin'),
    FlagDef('--resume', '-r', arg_type='session', description='Resume conversation by session ID'),
    FlagDef('--session-id', arg_type='string', description='Use specific session ID (UUID)'),
    FlagDef('--setting-sources', arg_type='string', description='Comma-separated list of setting sources'),
    FlagDef('--settings', arg_type='path', description='Path to settings JSON file or JSON string'),
    FlagDef('--strict-mcp-config', description='Only use MCP servers from --mcp-config'),
    FlagDef('--system-prompt', arg_type='string', description='System prompt for the session'),
    FlagDef('--tmux', description='Create tmux session for worktree'),
    FlagDef('--tools', arg_type='string', description='Specify available tools from built-in set'),
    FlagDef('--verbose', description='Override verbose mode setting'),
    FlagDef('--version', '-v', description='Output version number'),
    FlagDef('--worktree', '-w', arg_type='string', description='Create git worktree for session'),
    # Hidden flags (from binary analysis, not in --help)
    FlagDef(
        '--advisor', arg_type='model', documented=False, description='Server-side advisor tool with specified model'
    ),
    FlagDef('--agent-color', arg_type='string', documented=False, description='Teammate UI color'),
    FlagDef('--agent-id', arg_type='string', documented=False, description='Teammate agent ID'),
    FlagDef('--agent-name', arg_type='string', documented=False, description='Teammate display name'),
    FlagDef('--agent-type', arg_type='string', documented=False, description='Custom agent type for this teammate'),
    FlagDef(
        '--append-system-prompt-file', arg_type='path', documented=False, description='Append system prompt from file'
    ),
    FlagDef('--channels', arg_type='string', documented=False, description='MCP servers for channel notifications'),
    FlagDef('--cowork', documented=False, description='Use cowork_plugins directory'),
    FlagDef(
        '--dangerously-load-development-channels',
        arg_type='string',
        documented=False,
        description='Load channel servers not on the approved allowlist (dev only)',
    ),
    FlagDef(
        '--deep-link-cwd-b64', arg_type='string', documented=False, description='Base64url-encoded cwd (deep-link)'
    ),
    FlagDef('--deep-link-last-fetch', arg_type='string', documented=False, description='FETCH_HEAD mtime (deep-link)'),
    FlagDef('--deep-link-origin', documented=False, description='Session launched from a deep link'),
    FlagDef('--deep-link-repo', arg_type='string', documented=False, description='Deep-link ?repo= parameter'),
    FlagDef('--enable-auth-status', documented=False, description='Enable auth status messages in SDK mode'),
    FlagDef('--enable-auto-mode', documented=False, description='Opt in to auto mode'),
    FlagDef('--init', documented=False, description='Run Setup hooks with init trigger, then continue'),
    FlagDef('--init-only', documented=False, description='Run Setup and SessionStart:startup hooks, then exit'),
    FlagDef('--maintenance', documented=False, description='Run Setup hooks with maintenance trigger'),
    FlagDef(
        '--max-thinking-tokens', arg_type='string', documented=False, description='(DEPRECATED) Maximum thinking tokens'
    ),
    FlagDef('--max-turns', arg_type='string', documented=False, description='Maximum conversation turns'),
    FlagDef('--parent-session-id', arg_type='string', documented=False, description='Parent session ID for analytics'),
    FlagDef(
        '--permission-prompt-tool', arg_type='string', documented=False, description='MCP tool for permission prompts'
    ),
    FlagDef('--plan-mode-required', documented=False, description='Require plan mode before implementation'),
    FlagDef('--prefill', arg_type='string', documented=False, description='Pre-fill the prompt input with text'),
    FlagDef(
        '--prefill-b64', arg_type='string', documented=False, description='Base64url-encoded --prefill (deep-link)'
    ),
    FlagDef('--rc', arg_type='string', documented=False, description='Alias for --remote-control'),
    FlagDef('--remote', arg_type='string', documented=False, description='Create a remote session with description'),
    FlagDef(
        '--remote-control', arg_type='string', documented=False, description='Interactive session with Remote Control'
    ),
    FlagDef(
        '--resume-session-at', arg_type='string', documented=False, description='Resume only up to specified message ID'
    ),
    FlagDef(
        '--rewind-files',
        arg_type='string',
        documented=False,
        description='Restore files to state at user-message-id and exit',
    ),
    FlagDef('--sdk-url', arg_type='string', documented=False, description='Remote WebSocket endpoint for SDK I/O'),
    FlagDef('--session-mirror', documented=False, description='Emit transcript_mirror frames on stdout (SDK-internal)'),
    FlagDef('--system-prompt-file', arg_type='path', documented=False, description='System prompt from file'),
    FlagDef('--task-budget', arg_type='string', documented=False, description='API-side task budget in tokens'),
    FlagDef('--team-name', arg_type='string', documented=False, description='Team name for swarm coordination'),
    FlagDef(
        '--teammate-mode',
        arg_type='choice',
        choices=('auto', 'tmux', 'in-process'),
        documented=False,
        description='How to spawn teammates',
    ),
    FlagDef('--teleport', arg_type='session', documented=False, description='Resume a teleport session'),
    FlagDef(
        '--thinking',
        arg_type='choice',
        choices=('enabled', 'adaptive', 'disabled'),
        documented=False,
        description='Thinking mode',
    ),
    FlagDef(
        '--thinking-display',
        arg_type='choice',
        choices=('summarized', 'omitted'),
        documented=False,
        description='How thinking content appears in the response',
    ),
    FlagDef(
        '--workload', arg_type='string', documented=False, description='Workload tag for billing-header attribution'
    ),
]

# -- Subcommands ---------------------------------------------------------------

SUBCOMMANDS: Sequence[SubcommandDef] = [
    SubcommandDef('agents', 'List configured agents'),
    SubcommandDef('auth', 'Manage authentication'),
    SubcommandDef('auto-mode', 'Inspect auto mode classifier configuration'),
    SubcommandDef('doctor', 'Check health of auto-updater'),
    SubcommandDef('install', 'Install Claude Code native build'),
    SubcommandDef('mcp', 'Configure and manage MCP servers'),
    SubcommandDef('plugin', 'Manage Claude Code plugins'),
    SubcommandDef('setup-token', 'Set up long-lived authentication token'),
    SubcommandDef('update', 'Check for updates and install'),
]

MODEL_ALIASES: Sequence[str] = [
    'haiku',
    'opus',
    'sonnet',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-sonnet-4-6',
]
