"""Tests for the run-trailing-whitespace-and-newline-fix PostToolUse hook.

Canonical cases promoted from a 1020-case differential battery against upstream
``pre-commit/pre-commit-hooks`` (Layer 2). Each test illustrates one distinct
behavioral decision. Cases that surfaced upstream bugs or pathological corners
are memorialized in :class:`TestKnownLayer1Wins` so regressions in our binary
protection or BOM handling are caught.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType

import git
import pytest
from cc_lib.utils import temporary_module

REPO_ROOT = Path(git.Repo(__file__, search_parent_directories=True).working_tree_dir or '.').resolve(strict=True)
HOOK_SCRIPT = REPO_ROOT / 'hooks' / 'run-trailing-whitespace-and-newline-fix.py'

# Type aliases for fixture return types
FixBytesFn = Callable[[bytes, Path], bytes]
InTreeFileFactory = Callable[[str, bytes], Path]


@pytest.fixture(scope='session')
def hook_module() -> Generator[ModuleType]:
    """Import run-trailing-whitespace-and-newline-fix.py (hyphenated filename requires importlib)."""
    with temporary_module(HOOK_SCRIPT) as mod:
        yield mod


@pytest.fixture
def fix_bytes(hook_module: ModuleType) -> FixBytesFn:
    """The pure-function fix logic — input bytes + path, output normalized bytes."""
    fn: FixBytesFn = hook_module.fix_bytes
    return fn


# --- Pure-function tests for fix_bytes -----------------------------------------------


class TestCleanFiles:
    """A clean file passes through unchanged."""

    def test_clean_md(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'foo\n', Path('test.md')) == b'foo\n'

    def test_clean_yaml(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'key: value\n', Path('test.yaml')) == b'key: value\n'

    def test_empty(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'', Path('test.md')) == b''


class TestTrailingNewlineNormalization:
    """Exactly one trailing newline; collapse multiples, add if missing."""

    def test_adds_missing_newline(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'foo', Path('test.yaml')) == b'foo\n'

    def test_collapses_two_to_one(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'foo\n\n', Path('test.yaml')) == b'foo\n'

    def test_collapses_many_to_one(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'foo' + b'\n' * 10, Path('test.yaml')) == b'foo\n'

    def test_preserves_single_newline(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'foo\n', Path('test.yaml')) == b'foo\n'


class TestTrailingWhitespaceStripping:
    """Trailing space/tab stripped per line; leading whitespace preserved."""

    def test_strips_trailing_spaces(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'k: v  \n', Path('test.yaml')) == b'k: v\n'

    def test_strips_trailing_tabs(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'k: v\t\t\n', Path('test.yaml')) == b'k: v\n'

    def test_strips_mixed_space_tab(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'k: v \t \n', Path('test.yaml')) == b'k: v\n'

    def test_strips_form_feed_at_eof(self, fix_bytes: FixBytesFn) -> None:
        # Form feed (\x0c) and vertical tab (\x0b) are ASCII whitespace; upstream
        # treats them as such. We match.
        assert fix_bytes(b'foo\x0c\n', Path('test.md')) == b'foo\n'

    def test_preserves_leading_whitespace(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'    indented\n', Path('test.md')) == b'    indented\n'

    def test_preserves_internal_whitespace(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'a   b   c\n', Path('test.md')) == b'a   b   c\n'

    def test_strips_per_line(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'a  \nb  \nc\n', Path('test.yaml')) == b'a\nb\nc\n'


class TestMarkdownHardBreaks:
    """CommonMark hard-line-break syntax: exactly two trailing spaces preserved on .md/.markdown.

    Mirrors upstream's ``trailing-whitespace --markdown-linebreak-ext=md`` behavior.
    """

    def test_preserves_2_trailing_spaces(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'line  \nnext\n', Path('test.md')) == b'line  \nnext\n'

    def test_normalizes_3_spaces_to_2(self, fix_bytes: FixBytesFn) -> None:
        # 3+ trailing spaces is still a hard break per CommonMark; normalize to 2.
        assert fix_bytes(b'line   \nnext\n', Path('test.md')) == b'line  \nnext\n'

    def test_normalizes_many_spaces_to_2(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'line' + b' ' * 10 + b'\nnext\n', Path('test.md')) == b'line  \nnext\n'

    def test_strips_1_trailing_space(self, fix_bytes: FixBytesFn) -> None:
        # Single trailing space isn't a hard break; strip.
        assert fix_bytes(b'line \nnext\n', Path('test.md')) == b'line\nnext\n'

    def test_strips_trailing_tabs(self, fix_bytes: FixBytesFn) -> None:
        # Tabs are not the hard-break syntax even on markdown.
        assert fix_bytes(b'line\t\t\nnext\n', Path('test.md')) == b'line\nnext\n'

    def test_strips_tab_after_spaces(self, fix_bytes: FixBytesFn) -> None:
        # `  \t` — trailing tab means not the hard-break pattern; strip everything.
        assert fix_bytes(b'line  \t\nnext\n', Path('test.md')) == b'line\nnext\n'

    def test_keeps_2_spaces_after_tab(self, fix_bytes: FixBytesFn) -> None:
        # `\t  ` — line ends in 2 spaces; tab gets stripped, 2 spaces preserved.
        assert fix_bytes(b'line\t  \nnext\n', Path('test.md')) == b'line  \nnext\n'

    def test_blank_line_with_2_spaces_strips(self, fix_bytes: FixBytesFn) -> None:
        # A line containing only whitespace can't carry a hard break — strip it.
        assert fix_bytes(b'a\n  \nb\n', Path('test.md')) == b'a\n\nb\n'

    def test_markdown_ext_also_preserves(self, fix_bytes: FixBytesFn) -> None:
        # .markdown is also markdown (must match Layer 2's --markdown-linebreak-ext=markdown).
        assert fix_bytes(b'line  \nnext\n', Path('test.markdown')) == b'line  \nnext\n'

    def test_yaml_strips_2_spaces(self, fix_bytes: FixBytesFn) -> None:
        # Non-markdown extensions always strip trailing whitespace.
        assert fix_bytes(b'k: v  \n', Path('test.yaml')) == b'k: v\n'


class TestLineEndings:
    """Mid-content CRLF preserved; trailing CR normalized; final CRLF preserved if exact."""

    def test_preserves_mid_content_crlf(self, fix_bytes: FixBytesFn) -> None:
        # CRLF inside the file stays — the file's existing line-ending style is preserved.
        assert fix_bytes(b'foo\r\nbar\n', Path('test.md')) == b'foo\r\nbar\n'

    def test_preserves_final_crlf(self, fix_bytes: FixBytesFn) -> None:
        # File ending in exactly \r\n keeps it.
        assert fix_bytes(b'foo\nbar\r\n', Path('test.md')) == b'foo\nbar\r\n'

    def test_strips_trailing_cr_only(self, fix_bytes: FixBytesFn) -> None:
        # \n followed by stray \r at EOF — normalize.
        assert fix_bytes(b'foo\n\r', Path('test.md')) == b'foo\n'

    def test_preserves_crlf_with_extra_trailing_crlf(self, fix_bytes: FixBytesFn) -> None:
        # CRLF file with one blank line at EOF: collapse to single trailing CRLF
        # (matches upstream — Windows file's line-ending style is preserved).
        assert fix_bytes(b'foo\r\n\r\n', Path('test.md')) == b'foo\r\n'

    def test_preserves_crlf_with_many_trailing_crlf(self, fix_bytes: FixBytesFn) -> None:
        # CRLF file with many trailing blank lines: collapse to one.
        assert fix_bytes(b'foo\r\n\r\n\r\n', Path('test.md')) == b'foo\r\n'

    def test_preserves_crlf_with_mixed_trailing_lf(self, fix_bytes: FixBytesFn) -> None:
        # Trailing run starting with CRLF then LF: preserve CRLF style at EOF.
        assert fix_bytes(b'foo\r\n\n', Path('test.md')) == b'foo\r\n'

    def test_multiline_crlf_with_blank_at_end(self, fix_bytes: FixBytesFn) -> None:
        # Realistic Windows-authored file with trailing blank line.
        assert fix_bytes(b'line1\r\nline2\r\n\r\n', Path('test.md')) == b'line1\r\nline2\r\n'


class TestBOMHandling:
    """UTF-8 BOM is stripped after trailing-newline normalization (matches upstream chain)."""

    def test_strips_utf8_bom(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'\xef\xbb\xbf# Heading\n', Path('test.md')) == b'# Heading\n'

    def test_bom_only_normalizes_to_newline(self, fix_bytes: FixBytesFn) -> None:
        # File containing just a BOM: upstream's chain adds \n (via end-of-file-fixer)
        # before fix-byte-order-marker strips the BOM. Result is a single \n.
        # Note: this is non-idempotent — a second pass would truncate \n to ''. Matches L2.
        assert fix_bytes(b'\xef\xbb\xbf', Path('test.md')) == b'\n'


class TestAllWhitespaceTruncation:
    """Files that are entirely whitespace truncate to empty (matches upstream)."""

    def test_only_newline(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'\n', Path('test.md')) == b''

    def test_only_space(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b' ', Path('test.md')) == b''

    def test_only_tab(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'\t', Path('test.md')) == b''

    def test_many_newlines(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'\n\n\n\n\n', Path('test.md')) == b''

    def test_mixed_whitespace_lines(self, fix_bytes: FixBytesFn) -> None:
        assert fix_bytes(b'  \n  \n  \n', Path('test.md')) == b''


class TestIdempotency:
    """fix_bytes(fix_bytes(x)) == fix_bytes(x) on representative inputs."""

    @pytest.mark.parametrize(
        ('input_bytes', 'filename'),
        [
            (b'foo\n', 'test.md'),
            (b'foo', 'test.yaml'),
            (b'foo\n\n\n', 'test.md'),
            (b'k: v  \n', 'test.yaml'),
            (b'line one  \nline two\n', 'test.md'),
            (b'line one   \nline two\n', 'test.md'),
            (b'\xef\xbb\xbf# Heading\n', 'test.md'),
            (b'foo\r\nbar\n', 'test.md'),
            (b'foo\nbar\r\n', 'test.md'),
            (b'', 'test.md'),
        ],
        ids=[
            'clean-md',
            'no-eof-newline',
            'multiple-trailing-newlines',
            'yaml-trailing-ws',
            'md-hardbreak-2spaces',
            'md-3spaces-normalize',
            'utf8-bom',
            'crlf-mid',
            'crlf-at-eof',
            'empty',
        ],
    )
    def test_idempotent(self, fix_bytes: FixBytesFn, input_bytes: bytes, filename: str) -> None:
        once = fix_bytes(input_bytes, Path(filename))
        twice = fix_bytes(once, Path(filename))
        assert once == twice, f'non-idempotent: {input_bytes!r} → {once!r} → {twice!r}'


# --- Module-structure tests ----------------------------------------------------------


class TestModuleStructure:
    """Sanity checks on the hook module's API surface."""

    def test_hook_script_exists(self) -> None:
        assert HOOK_SCRIPT.is_file()

    def test_hook_script_is_executable(self) -> None:
        assert os.access(HOOK_SCRIPT, os.X_OK)

    def test_fix_bytes_is_callable(self, hook_module: ModuleType) -> None:
        assert callable(hook_module.fix_bytes)

    def test_blocked_extensions_includes_python(self, hook_module: ModuleType) -> None:
        # .py and .pyi go to ruff, not this hook
        assert '.py' in hook_module.BLOCKED_EXTENSIONS
        assert '.pyi' in hook_module.BLOCKED_EXTENSIONS

    def test_blocked_extensions_includes_csv_tsv(self, hook_module: ModuleType) -> None:
        # CSV/TSV legitimately use CRLF per RFC 4180
        assert '.csv' in hook_module.BLOCKED_EXTENSIONS
        assert '.tsv' in hook_module.BLOCKED_EXTENSIONS

    def test_blocked_extensions_includes_common_binary(self, hook_module: ModuleType) -> None:
        # A representative subset — full list documented in the script
        for ext in ('.png', '.jpg', '.pdf', '.zip', '.so', '.dylib', '.exe', '.ipynb'):
            assert ext in hook_module.BLOCKED_EXTENSIONS

    def test_markdown_extensions(self, hook_module: ModuleType) -> None:
        # Both extensions must be recognized so behavior matches Layer 2
        assert '.md' in hook_module.MARKDOWN_EXTENSIONS
        assert '.markdown' in hook_module.MARKDOWN_EXTENSIONS


# --- Integration tests (full hook flow via subprocess) -------------------------------


@pytest.fixture
def in_tree_file(tmp_path_factory: pytest.TempPathFactory) -> Generator[InTreeFileFactory]:
    """Create a writable file inside the repo (scope-check requires in-tree)."""
    # Use a scratch dir under REPO_ROOT — the hook's scope check requires the file
    # be under $CLAUDE_EXEC_LAUNCH_DIR. Namespace by xdist worker (PYTEST_XDIST_WORKER
    # is 'gw0'/'gw1'/… under -n, unset serially) so concurrent workers reusing the same
    # filenames (scratch.py, data.csv, …) can't clobber each other's files mid-test.
    worker = os.environ.get('PYTEST_XDIST_WORKER', 'serial')
    scratch_dir = REPO_ROOT / '.test-scratch' / worker
    scratch_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def make(name: str, content: bytes) -> Path:
        path = scratch_dir / name
        path.write_bytes(content)
        files.append(path)
        return path

    yield make
    for path in files:
        path.unlink(missing_ok=True)
    # rmdir the worker subdir, then the parent (no-op until the last worker empties it).
    for directory in (scratch_dir, scratch_dir.parent):
        try:
            directory.rmdir()
        except OSError:
            pass


class TestExtensionBlocklist:
    """The fast-path extension blocklist short-circuits before reading the file."""

    def test_python_file_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        original = b'def f():\n    pass  \n'  # has trailing whitespace
        path = in_tree_file('scratch.py', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original, '.py files are handled by run-ruff.py, not this hook'

    def test_csv_with_crlf_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        original = b'a,b\r\nc,d\r\n'  # legitimate RFC 4180 CRLF
        path = in_tree_file('data.csv', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original

    def test_png_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        original = b'\x89PNG\r\n\x1a\nfake\n'
        path = in_tree_file('image.png', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original


class TestBinaryContentSniff:
    """Content sniff catches binary files the extension blocklist misses.

    Null-byte test on first 8 KB classifies UTF-16 files masquerading as .md
    (and other extension-spoofed binaries) as binary, so we skip them.
    """

    def test_null_byte_at_start_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        original = b'\x00fake-binary-content\n'
        path = in_tree_file('binary.md', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original

    def test_null_byte_mid_file_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        original = b'header\x00body\n'
        path = in_tree_file('mixed.md', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original


class TestKnownLayer1Wins:
    """Memorialize cases where Layer 1 protects content that Layer 2 corrupts.

    Upstream pre-commit-hooks classifies files by extension via `identify` —
    a `.md` file containing UTF-16 or null bytes slips through and gets a raw
    `\\n` appended (corrupting the encoding). Our content-sniff catches these
    and skips them. Each test below verifies Layer 1 leaves the file ALONE.
    """

    def test_utf16_le_file_with_md_extension(self, in_tree_file: InTreeFileFactory) -> None:
        # A UTF-16 LE-encoded `.md` file should NOT be modified by our hook.
        # Upstream pre-commit-hooks would corrupt it because identify() classifies
        # by extension only; our content-sniff catches the null bytes.
        original = b'\xff\xfe' + 'hello\n'.encode('utf-16-le')
        path = in_tree_file('utf16.md', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original

    def test_only_null_byte_unchanged(self, in_tree_file: InTreeFileFactory) -> None:
        # A `.md` file containing a single null byte is binary; don't touch it.
        original = b'\x00'
        path = in_tree_file('null.md', original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original


class TestScopeGate:
    """Files outside $CLAUDE_EXEC_LAUNCH_DIR are silently skipped (matches run-ruff.py)."""

    def test_out_of_tree_file_unchanged(self, tmp_path: Path) -> None:
        # tmp_path is /tmp/... — outside the repo. Hook should no-op.
        original = b'foo  \n\n'  # would normally get normalized
        path = tmp_path / 'out.md'
        path.write_bytes(original)
        ec, after = _run_hook(path)
        assert ec == 0
        assert after == original

    def test_missing_file_no_error(self, in_tree_file: InTreeFileFactory, tmp_path: Path) -> None:
        # File path doesn't exist — exit cleanly without writing anything.
        nonexistent = REPO_ROOT / '.test-scratch' / 'does-not-exist.md'
        nonexistent.parent.mkdir(exist_ok=True)
        ec, after = _run_hook(nonexistent)
        assert ec == 0
        assert after == b''


# --- Private helpers ----------------------------------------------------------------


def _payload(file_path: Path) -> bytes:
    return json.dumps(
        {
            'session_id': 'x',
            'transcript_path': '/tmp/x',
            'cwd': str(REPO_ROOT),
            'tool_name': 'Edit',
            'tool_input': {'file_path': str(file_path)},
            'tool_use_id': 'x',
            'hook_event_name': 'PostToolUse',
        }
    ).encode()


def _run_hook(file_path: Path) -> tuple[int, bytes]:
    """Run the hook in a subprocess with CLAUDE_EXEC_LAUNCH_DIR pointed at the worktree."""
    env = {**os.environ, 'CLAUDE_EXEC_LAUNCH_DIR': str(REPO_ROOT)}
    proc = subprocess.run(
        [str(HOOK_SCRIPT)],
        input=_payload(file_path),
        capture_output=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=30,
        check=False,
    )
    after = file_path.read_bytes() if file_path.exists() else b''
    return proc.returncode, after
