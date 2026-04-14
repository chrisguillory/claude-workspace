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

Two-pass pipeline matching pre-commit structure:

1. ``ruff check --fix --fixable`` — lint + safe auto-fix
2. ``ruff format`` — AST-preserving formatting (never removes code/imports)

Auto-fixed by ruff check (--fixable whitelist)::

    Code | Rule                           | Why safe         | Ref
    ---- | ------------------------------ | ---------------- | ---
    I001 | unsorted-imports               | Mechanical order | https://docs.astral.sh/ruff/rules/unsorted-imports/
    W291 | trailing-whitespace            | Invisible chars  | https://docs.astral.sh/ruff/rules/trailing-whitespace/
    W292 | missing-newline-at-end-of-file | POSIX convention | https://docs.astral.sh/ruff/rules/missing-newline-at-end-of-file/
    W293 | blank-line-with-whitespace     | Blank line spaces | https://docs.astral.sh/ruff/rules/blank-line-with-whitespace/

Auto-fixed by ruff format:
    Line length wrapping, quote normalization, trailing commas,
    parenthesization. W291/W292/W293 overlap is harmless.

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

    # Pass 1: check all project rules, auto-fix only the safe formatting ones.
    # --fixable restricts which rules --fix is allowed to touch.
    subprocess.run(
        ['ruff', 'check', '--fix', '--fixable', 'I001,W291,W292,W293', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    # Pass 2: format (AST-preserving — never removes code/imports)
    subprocess.run(
        ['ruff', 'format', file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    return 0


@boundary.handler(subprocess.CalledProcessError)
def _handle_subprocess(exc: subprocess.CalledProcessError) -> None:
    sys.stderr.buffer.write(exc.stdout)


if __name__ == '__main__':
    sys.exit(main())
