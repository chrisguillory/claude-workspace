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

Exit codes (Claude Code hook protocol):
    0 — Success, no feedback shown.
    2 — Blocking error: stderr is shown to Claude as inline feedback
        and Claude must address the violation before continuing.

ErrorBoundary guarantees exit 2 on any unhandled exception (e.g.,
Pydantic ValidationError from schema drift). Exit 1 is never used —
it is a black hole where the model sees nothing.
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

    # Only lint Python files.
    if not file_path.endswith('.py'):
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
