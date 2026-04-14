#!/usr/bin/env -S uv run --quiet --script
"""Inject session environment into Bash tool subprocess environments.

Uses the CLAUDE_ENV_FILE mechanism to export variables into every subsequent
Bash tool command for the session's lifetime. Currently injects:

    CLAUDE_CODE_SESSION_ID  Session identifier. Claude Code sets this for
        Anthropic employees via a build-time USER_TYPE=ant gate in Shell.ts
        (dead-code-eliminated from the public binary). This hook replicates
        it for all users.

Source evidence:
    Shell.ts:323-327 — ant-only CLAUDE_CODE_SESSION_ID injection (DCE'd)
    hooks.ts:925 — CLAUDE_ENV_FILE passed to SessionStart hooks (not gated)
    bashProvider.ts:170-173 — session env script sourced before commands
    AsyncHookRegistry.ts:259-261 — cache invalidated after hook completes

See: https://code.claude.com/docs/en/hooks#sessionstart
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc_lib",
# ]
#
# [tool.uv.sources]
# cc_lib = { path = "../cc-lib/", editable = true }
# ///
from __future__ import annotations

import os
import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import SessionStartHookInput
from cc_lib.utils import get_claude_config_home_dir

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> None:
    hook_data = SessionStartHookInput.model_validate_json(sys.stdin.read())
    content = f'export CLAUDE_CODE_SESSION_ID={hook_data.session_id}\n'

    # Write to CLAUDE_ENV_FILE if available (works on startup)
    env_file = os.environ.get('CLAUDE_ENV_FILE')
    if env_file:
        Path(env_file).parent.mkdir(parents=True, exist_ok=True)
        Path(env_file).write_text(content)

    # Also write to the path the Bash tool actually reads on resume.
    # CLAUDE_ENV_FILE points to the boot-time session ID directory,
    # but after switchSession() the Bash tool reads from the resumed
    # session ID directory (Claude Code bug: REPL.tsx fires hooks at
    # line 1782 BEFORE switchSession() at line 1846).
    session_env_dir = get_claude_config_home_dir() / 'session-env' / hook_data.session_id
    filename = Path(env_file).name if env_file else 'sessionstart-hook-1.sh'
    target = session_env_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    print(f'Injected CLAUDE_CODE_SESSION_ID={hook_data.session_id}')


if __name__ == '__main__':
    main()
