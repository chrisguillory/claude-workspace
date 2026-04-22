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

import subprocess
import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.exceptions import HookTreeMismatchError
from cc_lib.schemas.hooks import PostToolUseHookInput
from cc_lib.utils import validate_hook_tree

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: run-linter.py <linter-path> [extra-args...]', file=sys.stderr)
        return 2

    launch_dir = validate_hook_tree(Path(__file__))

    payload = PostToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    file_path = payload.tool_input.get('file_path', '')

    # File-type gate: Python sources only (.py, .pyi).
    if not file_path.endswith(('.py', '.pyi')):
        return 0

    # Skip if the file doesn't exist (e.g., deleted).
    file = Path(file_path)
    if not file.is_file():
        return 0

    # Scope: silently skip files outside the project.
    if not file.resolve().is_relative_to(launch_dir):
        return 0

    # Run with cwd=launch_dir so relative linter paths resolve correctly and
    # Python adds the linter's directory to sys.path (for _lib imports).
    # Claude Code's hook subprocess cwd tracks STATE.cwd (drifts with user's
    # `cd` commands), so inherited cwd isn't reliable.
    subprocess.run(
        [sys.executable, sys.argv[1], *sys.argv[2:], file_path],
        cwd=launch_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    return 0


@boundary.handler(subprocess.CalledProcessError)
def _handle_subprocess(exc: subprocess.CalledProcessError) -> None:
    sys.stderr.buffer.write(exc.stdout)


@boundary.handler(HookTreeMismatchError)
def _handle_tree_mismatch(exc: HookTreeMismatchError) -> None:
    print(str(exc), file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
