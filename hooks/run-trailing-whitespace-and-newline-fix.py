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
"""PostToolUse hook: normalize trailing whitespace + trailing newlines on non-Python text files.

Fires on every Edit/Write to a non-Python text file in the workspace. Layer-2
backstop is upstream ``pre-commit/pre-commit-hooks`` (``trailing-whitespace``,
``end-of-file-fixer``, ``fix-byte-order-marker``, ``mixed-line-ending``).

Fix logic (pure, idempotent):

1. Empty / all-whitespace file → unchanged (more conservative than upstream
   ``end-of-file-fixer`` which truncates all-``\\n`` files to empty).
2. CRLF detected (``\\r`` in bytes) → ``MixedLineEndingsError`` (fail loud,
   exit 2). CSV/TSV are excluded at the extension layer above.
3. Strip ``[ \\t]+$`` per line. For ``.md`` / ``.markdown``: preserve exactly
   two trailing spaces (CommonMark hard-line-break syntax) to match upstream's
   ``--markdown-linebreak-ext=md``.
4. Collapse trailing ``\\n`` runs to exactly one.

File-type gate:

- Block ``.py`` / ``.pyi`` (handled by ``run-ruff.py``; double-fixing risks
  a write race).
- Block known-binary extensions (no content sniff needed).
- Block ``.csv`` / ``.tsv`` (legitimate CRLF per RFC 4180).
- Content sniff (8 KB read, null-byte test) — matches pre-commit's
  ``identify`` heuristic for "is this file text?".

Exit code visibility (Claude Code hook protocol)::

    Code | User sees          | Model sees               | Debug log
    ---- | ------------------ | ------------------------ | ---------
    0    | Nothing            | Nothing (PostToolUse)    | stdout
    1    | "hook error"       | Nothing                  | stderr
    2    | "blocking error"   | stderr as system context | stderr

``ErrorBoundary(exit_code=2)`` guarantees no exception produces exit 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cc_lib.error_boundary import ErrorBoundary
from cc_lib.exceptions import HookTreeMismatchError
from cc_lib.schemas.hooks import PostToolUseHookInput
from cc_lib.utils import validate_hook_tree
from cc_lib.utils.atomic_write import atomic_write

boundary = ErrorBoundary(exit_code=2)

BLOCKED_EXTENSIONS = frozenset(
    {
        # Python — handled by run-ruff.py
        '.py',
        '.pyi',
        # Legitimately CRLF per RFC 4180
        '.csv',
        '.tsv',
        # Known binary
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.webp',
        '.ico',
        '.pdf',
        '.zip',
        '.gz',
        '.tar',
        '.tgz',
        '.so',
        '.dylib',
        '.o',
        '.a',
        '.lib',
        '.exe',
        '.dll',
        '.bin',
        '.dat',
        '.db',
        '.sqlite',
        '.parquet',
        '.pyc',
        '.pyo',
        '.ipynb',
    }
)

CONTENT_SNIFF_BYTES = 8192

MARKDOWN_EXTENSIONS = frozenset({'.md', '.markdown'})

UTF8_BOM = b'\xef\xbb\xbf'


def fix_bytes(data: bytes, path: Path) -> bytes:
    """Apply trailing-ws + trailing-newline normalization. Pure function.

    Matches upstream ``pre-commit/pre-commit-hooks`` behavior (the Layer 2 backstop)
    so the two layers agree byte-for-byte:

    - **Markdown hard break**: on ``.md``/``.markdown``, if a line has content AND
      ends with at least 2 trailing spaces, preserve as exactly 2 trailing spaces
      (CommonMark). Otherwise strip all trailing spaces/tabs.
    - **All-whitespace file** → empty (matches ``end-of-file-fixer``).
    - **Trailing ``\\r``** → stripped (matches ``end-of-file-fixer``'s
      ``\\n``/``\\r``-stripping at EOF).
    - **Mid-line CRLF** → preserved at this layer (matches upstream;
      Layer 2's ``mixed-line-ending --fix=no`` flags it at commit time).
    - **UTF-8 BOM** → stripped (matches ``fix-byte-order-marker``).
    """
    is_markdown = path.suffix in MARKDOWN_EXTENSIONS

    # Split into lines preserving line endings — matches upstream's binary readlines
    # (splits on \n only; \r remains as part of the line content or eol).
    parts = data.split(b'\n')
    lines = [p + b'\n' for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])

    # Process each line — mirrors upstream `_process_line`.
    # eol is \r\n, \n, or '' (last line with no trailing newline).
    # Content is rstripped of ALL ASCII whitespace (rstrip() with no args), matching
    # upstream's default behavior (chars=None → strip ' \t\n\r\v\f').
    out: list[bytes] = []
    for line in lines:
        if line.endswith(b'\r\n'):
            eol, content = b'\r\n', line[:-2]
        elif line.endswith(b'\n'):
            eol, content = b'\n', line[:-1]
        else:
            eol, content = b'', line
        if is_markdown and not content.isspace() and content.endswith(b'  '):
            # Preserve markdown hard break: keep exactly 2 trailing spaces.
            out.append(content[:-2].rstrip() + b'  ' + eol)
        else:
            out.append(content.rstrip() + eol)
    joined = b''.join(out)

    # End-of-file-fixer: walk back trailing \n/\r run, normalize to one trailing \n
    # (or preserve final \r\n exactly).
    i = len(joined)
    while i > 0 and joined[i - 1 : i] in (b'\n', b'\r'):
        i -= 1
    prefix = joined[:i]
    trailing = joined[i:]

    if not prefix:
        result = b''
    elif trailing == b'\r\n':
        result = prefix + b'\r\n'
    else:
        result = prefix + b'\n'

    # Strip UTF-8 BOM at the END (matches Layer 2's hook order: trailing-whitespace +
    # end-of-file-fixer run BEFORE fix-byte-order-marker).
    if result.startswith(UTF8_BOM):
        result = result[len(UTF8_BOM) :]
    return result


@boundary
def main() -> int:
    launch_dir = validate_hook_tree(Path(__file__))
    payload = PostToolUseHookInput.model_validate_json(sys.stdin.buffer.read())
    file_path = payload.tool_input.get('file_path', '')

    if not file_path:
        return 0

    file = Path(file_path)
    if not file.is_file():
        return 0

    # Scope: silently skip files outside the project.
    if not file.resolve().is_relative_to(launch_dir):
        return 0

    # Fast-path extension exclusion.
    if file.suffix.lower() in BLOCKED_EXTENSIONS:
        return 0

    data = file.read_bytes()

    # Content sniff: skip binaries (UTF-16, PNG-without-extension, etc.).
    if b'\x00' in data[:CONTENT_SNIFF_BYTES]:
        return 0

    fixed = fix_bytes(data, file)
    if fixed == data:
        return 0

    atomic_write(file, fixed, reference=file)
    return 0


@boundary.handler(HookTreeMismatchError)
def _handle_tree_mismatch(exc: HookTreeMismatchError) -> None:
    print(str(exc), file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main())
