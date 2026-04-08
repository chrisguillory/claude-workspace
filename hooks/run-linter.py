#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../cc-lib/", editable = true }
# ///
"""Bridge between Claude Code PostToolUse hooks and linters.

Reads tool context from stdin (JSON), extracts the edited file path,
and runs the specified linter on that file. If the linter finds
violations, exits 2 with the linter output on stderr — this is the
most reliable way to surface feedback to Claude Code.

Usage in .claude/settings.json::

    {
      "hooks": {
        "PostToolUse": [{
          "matcher": "Edit|Write",
          "hooks": [{
            "type": "command",
            "command": "hooks/run-linter.py linters/strict_typing_linter.py"
          }]
        }]
      }
    }

The linter path is relative to the repository root (cwd of the hook).
Multiple linters can be configured as separate hook entries.

Exit code visibility (Claude Code hook protocol)::

    Code | User sees          | Model sees               | Debug log
    ---- | ------------------ | ------------------------ | ---------
    0    | Nothing            | Nothing (PostToolUse)    | stdout
    1    | "hook error"       | Nothing                  | stderr
    2    | "blocking error"   | stderr as system context | stderr

Exit 1 is a black hole — the model cannot see the error or self-diagnose.
Hooks should never exit 1. Use exit 2 to guarantee all errors (including
unexpected exceptions) surface to the model as actionable feedback.

ErrorBoundary(exit_code=2) enforces this: any unhandled exception (e.g.,
Pydantic ValidationError from schema drift) exits 2 with the traceback
on stderr, which the model sees and can act on.

Ref: https://code.claude.com/docs/en/hooks (exit code semantics)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import PostToolUseHookInput

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: run-linter.py <linter-script> [extra-args...]', file=sys.stderr)
        return 2

    linter = sys.argv[1]
    extra_args = sys.argv[2:]

    payload = PostToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    file_path = payload.tool_input.get('file_path', '')

    # File-type gate: Python sources only (.py, .pyi).
    # This lives here — not in the settings.json `if` pattern — because Claude Code's
    # hook `if` uses a custom regex matcher (not glob): `*` becomes `.*`, but brace
    # expansion, character classes, and extglob are all escaped to literals.
    # Matching `.py` + `.pyi` would require duplicate hook entries per extension.
    # Centralizing here keeps settings.json clean and supports both extensions.
    if not file_path.endswith(('.py', '.pyi')):
        return 0

    # Skip if the file doesn't exist (e.g., deleted).
    if not Path(file_path).is_file():
        return 0

    # Ensure _lib subpackage (config, hashability_inspector) is importable.
    linter_dir = str(Path(linter).resolve().parent)
    env = dict(os.environ)
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{linter_dir}:{existing}' if existing else linter_dir

    result = subprocess.run(
        [sys.executable, linter, *extra_args, file_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.returncode != 0:
        sys.stderr.buffer.write(result.stdout)
        return 2

    return 0


if __name__ == '__main__':
    sys.exit(main())
