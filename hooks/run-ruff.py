#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "cc-lib",
# ]
#
# [tool.uv.sources]
# cc-lib = { path = "../cc-lib/", editable = true }
# ///
"""PostToolUse hook: run ruff on edited Python files.

Runs all project rules but only auto-fixes mechanical formatting via
``--fix --fixable``. Real violations are surfaced to the model via exit 2.

Auto-fixed rules (--fixable whitelist)::

    Code | Rule                  | Why safe           | Ref
    ---- | --------------------- | ------------------ | ---
    I001 | unsorted-imports      | Mechanical order   | https://docs.astral.sh/ruff/rules/unsorted-imports/
    W291 | trailing-whitespace   | Invisible chars    | https://docs.astral.sh/ruff/rules/trailing-whitespace/
    W292 | no-newline-at-eof     | POSIX convention   | https://docs.astral.sh/ruff/rules/missing-newline-at-end-of-file/
    W293 | whitespace-with-indent| Blank line spaces  | https://docs.astral.sh/ruff/rules/blank-line-with-whitespace/

NOT auto-fixed (surfaced as errors for the model to fix):
    F401 (unused-import) — would silently remove imports that may be
    intentionally staged for upcoming code.

Exit code visibility (Claude Code hook protocol)::

    Code | User sees          | Model sees               | Debug log
    ---- | ------------------ | ------------------------ | ---------
    0    | Nothing            | Nothing (PostToolUse)    | stdout
    1    | "hook error"       | Nothing                  | stderr
    2    | "blocking error"   | stderr as system context | stderr

ErrorBoundary(exit_code=2) guarantees no exception produces exit 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.schemas.hooks import PostToolUseHookInput

boundary = ErrorBoundary(exit_code=2)


@boundary
def main() -> int:
    payload = PostToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    file_path = payload.tool_input.get('file_path', '')

    if not file_path.endswith(('.py', '.pyi')):
        return 0

    if not Path(file_path).is_file():
        return 0

    # Single pass: check all project rules, auto-fix only the safe formatting ones.
    # --fixable restricts which rules --fix is allowed to touch.
    result = subprocess.run(
        ['ruff', 'check', '--fix', '--fixable', 'I001,W291,W292,W293', file_path],
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
