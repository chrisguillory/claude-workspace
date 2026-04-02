#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Suppression rationale linter — requires explanations on lint suppression directives.

AI models (and humans under time pressure) add suppression directives like
``# noqa: E501`` or ``# type: ignore[override]`` without explaining WHY the
suppression is justified. This hides technical debt and makes code review
impossible — you cannot distinguish a justified suppression from a lazy one
without context.

For AI-generated code specifically, requiring a rationale forces the model to
articulate its reasoning, which often reveals that the suppression is unjustified
and the underlying issue should be fixed instead.

Motivation:
    No existing Python tool enforces rationale on suppression directives.
    Ruff issue #5182 has been open since June 2023 with no implementation
    timeline ("needs-decision" label, stalled nearly 3 years). Other
    ecosystems have solved this:

    - Go: golangci-lint nolintlint — ``require-explanation: true``
    - JavaScript: eslint-plugin-eslint-comments — ``require-description`` rule
    - Rust: clippy ``allow_attributes_without_reason`` restriction lint

    Beyond presence, ~40% of mandatory rationales are useless placeholder text
    (Biome community data, 14,600+ files). SUP005/SUP006 address this with
    heuristic blocklists that catch ~70% of bad rationales at zero cost — no
    ML model needed. Research shows best ML classifiers for comment quality hit
    only 69% accuracy (Oztas et al., 2025), so heuristics outperform ML here.

Prior Art & References:
    - ruff #5182: https://github.com/astral-sh/ruff/issues/5182
    - ruff #8453 (duplicate): https://github.com/astral-sh/ruff/issues/8453
    - Go nolintlint: https://github.com/ashanbrown/nolintlint
    - ESLint require-description: https://eslint-community.github.io/eslint-plugin-eslint-comments/rules/require-description.html
    - Rust clippy: https://github.com/rust-lang/rust-clippy/issues/8502
    - Dan Abramov, "Suppressions of Suppressions": https://overreacted.io/suppressions-of-suppressions/
    - FSE 2025 suppression study: https://dl.acm.org/doi/10.1145/3715729
    - Oztas et al. 2025, comment smell detection: https://arxiv.org/abs/2504.18956

Retirement:
    If ruff implements #5182, evaluate whether the native rule covers all six
    rule codes and both presence + quality checking. If so, this linter can be
    retired. Monitor the issue for movement.

Rules:
    SUP001 noqa-without-rationale          ``# noqa`` directive without explanation
    SUP002 type-ignore-without-rationale   ``# type: ignore`` without explanation
    SUP003 pyright-ignore-without-rationale ``# pyright: ignore`` without explanation
    SUP004 custom-suppress-without-rationale Custom linter suppression without explanation
    SUP005 vague-rationale                  Rationale is present but uninformative
    SUP006 tautological-rationale           Rationale merely restates the error description

Rationale format:
    Suppression directives must include explanatory text after a separator.
    Accepted separators: ``—`` (em dash, preferred), ``--``, ``---``, ``#``.

    Valid examples:
        x = 1  # noqa: F841 — assigned for side effect in test setup
        x = 1  # noqa: F841 -- assigned for side effect in test setup
        x = 1  # type: ignore[override] — covariant return is safe here
        x = 1  # strict_typing_linter.py: mutable-type — pydantic requires list

    Invalid examples:
        x = 1  # noqa: F841                           (no rationale)
        x = 1  # noqa: F841 — intentional              (vague, SUP005)
        x = 1  # noqa: F841 — unused variable          (tautological, SUP006)
        x = 1  # noqa: F841 TODO fix later             (no separator)

Escape hatches:
    # suppression_rationale_linter.py: skip-file
    # suppression_rationale_linter.py: <rule-code> — <reason>

Design Philosophy:
    - Line-based scanner, not AST: suppression directives are comments, and
      AST parsing strips comments. We need the raw text.
    - Comment-first architecture: find the comment boundary once per line, then
      only search within the comment. This eliminates an entire class of false
      positives (directives inside string literals) by construction.
    - Heuristics over ML: blocklist patterns catch the majority of bad rationales
      at zero runtime cost. LLM-based validation is deferred until accuracy and
      volume justify the cost (see research references above).
    - Error-only, no auto-fix: forces conscious decision at each occurrence.
    - Standalone: no external dependencies, works with any Python 3.13+ install.

Exit codes:
    0 — No violations found
    1 — Violations found

Known Limitations:
    - Multi-line string detection is heuristic (triple-quote counting). Edge
      cases with escaped quotes or mixed delimiters may produce false results.
    - Does not check suppression directives in non-Python files (YAML, TOML).
    - SUP005/SUP006 quality checks use pattern matching, not semantic analysis.
      A rationale like "needed for X" passes even if X is wrong. Code review
      catches semantically incorrect rationales; this linter catches missing
      and obviously low-effort ones.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from _lib.config import find_config, find_python_files, get_per_file_ignored_codes, load_per_file_ignores

# -- Configuration ------------------------------------------------------------

DIRECTIVE_PREFIX = '# suppression_rationale_linter.py:'

DEFAULT_CUSTOM_PREFIXES: Sequence[str] = (
    '# exception_safety_linter.py:',
    '# reexport_linter.py:',
    '# strict_typing_linter.py:',
)

NOQA_RE = re.compile(
    r'#\s*noqa\b'
    r'(?:\s*:\s*[A-Z0-9]+'
    r'(?:\s*,\s*[A-Z0-9]+)*)?'
    r'(?P<after>.*)',
    re.IGNORECASE,
)

TYPE_IGNORE_RE = re.compile(
    r'#\s*type\s*:\s*ignore\b'
    r'(?:\s*\[[^\]]*\])?'
    r'(?P<after>.*)',
    re.IGNORECASE,
)

PYRIGHT_IGNORE_RE = re.compile(
    r'#\s*pyright\s*:\s*ignore\b'
    r'(?:\s*\[[^\]]*\])?'
    r'(?P<after>.*)',
    re.IGNORECASE,
)


def build_custom_prefix_re(prefix: str) -> re.Pattern[str]:
    """Build regex for a custom linter directive prefix."""
    escaped = re.escape(prefix)
    return re.compile(
        escaped + r'\s*[a-z0-9][-a-z0-9]*' + r'(?:\s*,\s*[a-z0-9][-a-z0-9]*)*' + r'(?P<after>.*)',
        re.IGNORECASE,
    )


# -- Data Types ---------------------------------------------------------------

type ErrorCode = str

type ViolationKind = Literal[
    'noqa-without-rationale',
    'type-ignore-without-rationale',
    'pyright-ignore-without-rationale',
    'custom-suppress-without-rationale',
    'vague-rationale',
    'tautological-rationale',
]

ERROR_CODES: Mapping[ViolationKind, ErrorCode] = {
    'noqa-without-rationale': 'SUP001',
    'type-ignore-without-rationale': 'SUP002',
    'pyright-ignore-without-rationale': 'SUP003',
    'custom-suppress-without-rationale': 'SUP004',
    'vague-rationale': 'SUP005',
    'tautological-rationale': 'SUP006',
}

VIOLATION_MESSAGES: Mapping[ViolationKind, str] = {
    'noqa-without-rationale': '`# noqa` directive without rationale',
    'type-ignore-without-rationale': '`# type: ignore` directive without rationale',
    'pyright-ignore-without-rationale': '`# pyright: ignore` directive without rationale',
    'custom-suppress-without-rationale': 'Custom linter suppression without rationale',
    'vague-rationale': 'Suppression rationale is too vague to be informative',
    'tautological-rationale': 'Suppression rationale merely restates the error description',
}

FIX_SUGGESTIONS: Mapping[ViolationKind, str] = {
    'noqa-without-rationale': "Add ' — <reason>' after the code, e.g.: # noqa: F841 — assigned for side effect",
    'type-ignore-without-rationale': "Add ' — <reason>' after the ignore, e.g.: # type: ignore[override] — covariant return",
    'pyright-ignore-without-rationale': "Add ' — <reason>' after the ignore, e.g.: # pyright: ignore[...] — stub mismatch",
    'custom-suppress-without-rationale': "Add ' — <reason>' after the code, e.g.: # my_linter.py: rule — justified because ...",
    'vague-rationale': "Replace with a specific explanation of WHY the suppression is needed, not just 'intentional' or 'needed'",
    'tautological-rationale': "Explain WHY you're suppressing, not WHAT the error is. 'unused variable' restates F841; say what it's for",
}


@dataclass(frozen=True)
class Violation:
    """A detected suppression rationale violation."""

    filepath: Path
    line: int
    column: int
    kind: ViolationKind
    source_line: str
    directive_text: str


# -- Main Entry Point ---------------------------------------------------------


def main() -> int:
    """CLI entry point: parse args, collect files, check, report."""
    args = parse_args()
    exclude_dirs = set(args.exclude) | {'.venv', 'venv', '__pycache__', '.git'}
    ignored_kinds: set[ViolationKind] = set(args.ignore)
    respect_gitignore = not args.no_gitignore

    if args.no_custom:
        custom_prefixes: Sequence[str] = ()
    elif args.custom_prefix is not None:
        custom_prefixes = args.custom_prefix
    else:
        custom_prefixes = DEFAULT_CUSTOM_PREFIXES

    # Collect files
    files: list[Path] = []
    for arg in args.paths:
        path = Path(arg)
        if arg == '.' or path.is_dir():
            files.extend(find_python_files(path, exclude_dirs, respect_gitignore))
        elif path.suffix == '.py' and path.is_file():
            files.append(path)

    if not files:
        return 0

    # Resolve config path once if --config was given
    explicit_config = Path(args.config) if args.config else None

    # Check all files
    all_violations: list[Violation] = []
    for filepath in files:
        # Per-file-ignores from pyproject.toml
        per_file_codes: Set[str] = set()
        skip_file_via_config = False
        if not args.no_config:
            if explicit_config is not None:
                config_path = explicit_config
                project_root = explicit_config.parent
            else:
                result = find_config(filepath, 'suppression-rationale-linter')
                if result is not None:
                    config_path, project_root = result
                else:
                    config_path = None
                    project_root = None

            if config_path is not None and project_root is not None:
                per_file_ignores = load_per_file_ignores('suppression-rationale-linter', config_path)
                per_file_codes = get_per_file_ignored_codes(filepath, per_file_ignores, project_root)
                if 'skip-file' in per_file_codes:
                    skip_file_via_config = True

        if skip_file_via_config:
            continue

        try:
            violations = check_file(
                filepath,
                custom_prefixes,
                respect_skip_file=not args.no_skip_file,
            )

            # Filter by per-file ignored codes (codes are violation kinds directly)
            if per_file_codes:
                violations = [v for v in violations if v.kind not in per_file_codes]

            all_violations.extend(violations)
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f'{filepath}: {e}', file=sys.stderr)

    # Filter ignored kinds
    if ignored_kinds:
        all_violations = [v for v in all_violations if v.kind not in ignored_kinds]

    # Report
    if all_violations:
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line)):
            print(format_violation(v))
            print()

        file_count = len({v.filepath for v in all_violations})
        print(f'Found {len(all_violations)} violation(s) in {file_count} file(s).')
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Check that lint suppression directives include a rationale.',
        epilog=(
            'Checks # noqa, # type: ignore, # pyright: ignore, and custom linter '
            'directives for explanatory text after a separator (\u2014 or --).'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='*',
        default=['.'],
        help='Files or directories to check (default: current directory)',
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        metavar='DIR',
        help='Directories to exclude when searching recursively',
    )
    parser.add_argument(
        '--ignore',
        nargs='*',
        default=[],
        metavar='CODE',
        choices=list(ERROR_CODES.keys()),
        help='Violation codes to ignore globally',
    )
    parser.add_argument(
        '--custom-prefix',
        nargs='*',
        default=None,
        metavar='PREFIX',
        help='Custom linter directive prefixes to check (default: strict_typing + exception_safety)',
    )
    parser.add_argument(
        '--no-custom',
        action='store_true',
        help='Disable checking custom linter directive prefixes (SUP004)',
    )
    parser.add_argument(
        '--no-gitignore',
        action='store_true',
        help='Do not respect .gitignore when scanning directories',
    )
    parser.add_argument(
        '--no-skip-file',
        action='store_true',
        help='Ignore skip-file directives (used by validation harnesses)',
    )
    parser.add_argument(
        '--config',
        default=None,
        metavar='PATH',
        help='Path to pyproject.toml (default: auto-discover by walking up from each file)',
    )
    parser.add_argument(
        '--no-config',
        action='store_true',
        help='Disable reading per-file-ignores from pyproject.toml',
    )
    return parser.parse_args()


# -- File Processing ----------------------------------------------------------


def check_file(
    filepath: Path,
    custom_prefixes: Sequence[str],
    *,
    respect_skip_file: bool = True,
) -> Sequence[Violation]:
    """Check a single file for suppression directives missing rationale."""
    source = filepath.read_text(encoding='utf-8')
    lines = source.splitlines()

    # File-level skip directive (check first 10 lines)
    if respect_skip_file:
        prefix_lower = DIRECTIVE_PREFIX.lower()
        for line in lines[:10]:
            if prefix_lower in line.lower() and 'skip-file' in line.lower():
                return []

    multiline_lines = track_multiline_strings(lines)
    scanner = SuppressionScanner(filepath, custom_prefixes)

    for i, line in enumerate(lines):
        lineno = i + 1
        if lineno in multiline_lines:
            continue
        if not line.strip():
            continue
        scanner.check_line(lineno, line)

    return scanner.violations


# -- Line Scanning ------------------------------------------------------------

# Standard directive patterns paired with their violation kind.
# The scanner iterates this table instead of having per-directive methods.
STANDARD_CHECKS: Sequence[tuple[re.Pattern[str], ViolationKind]] = [
    (NOQA_RE, 'noqa-without-rationale'),
    (TYPE_IGNORE_RE, 'type-ignore-without-rationale'),
    (PYRIGHT_IGNORE_RE, 'pyright-ignore-without-rationale'),
]


class RationaleChecker:
    """Validates suppression rationale presence and quality.

    Assessment pipeline:
    1. Presence — does a rationale exist after a separator? (SUP001-004)
    2. Vague — is the rationale too generic to be informative? (SUP005)
    3. Tautological — does it merely restate the error description? (SUP006)

    ~70% of bad rationales are catchable by pattern matching (Oztas et al., 2025).
    LLM-based quality validation is deferred until accuracy justifies the cost.
    """

    MIN_LENGTH = 2

    SEPARATORS = ('---', '--', '\u2014', '\u2013', '#')

    VAGUE_RE = re.compile(
        r'^('
        r'intentional|needed|required|necessary|'
        r'ok|okay|fine|safe|valid|correct|expected|'
        r'(safe|ok|okay|fine|needed|required) here|'
        r'because|'
        r'(this is )?(intentional|needed|ok|okay|fine|safe)|'
        r'(ignore|skip|suppress) (this|it)|'
        r'not applicable|n/a|na|todo|fixme|hack|workaround'
        r')\.?$',
        re.IGNORECASE,
    )

    TAUTOLOGICAL: Mapping[str, Sequence[str]] = {
        'F841': ['unused variable', 'variable is unused', 'variable not used', 'never used'],
        'F401': ['unused import', 'import is unused', 'import not used'],
        'E501': ['line too long', 'line length', 'long line'],
        'E722': ['bare except', 'broad except'],
        'B006': ['mutable default', 'default argument', 'mutable argument'],
        'B008': ['function call in default', 'call in default argument'],
        'S101': ['use of assert', 'assert detected', 'assert statement'],
        'PLC0415': ['import outside top level', 'import not at top'],
    }

    @classmethod
    def _find_after_separator(cls, after_text: str) -> str:
        """Extract text after the first recognized separator."""
        text = after_text.strip()
        if not text:
            return ''

        for sep in cls.SEPARATORS:
            idx = text.find(sep)
            if idx != -1:
                return text[idx + len(sep) :].strip()

        # Single hyphen requires surrounding spaces to avoid matching codes like E501-x
        single_hyphen_idx = text.find(' - ')
        if single_hyphen_idx == -1 and text.startswith('- '):
            single_hyphen_idx = 0
        if single_hyphen_idx != -1:
            return text[single_hyphen_idx + 3 :].strip() if single_hyphen_idx > 0 else text[2:].strip()

        return ''

    @classmethod
    def has_rationale(cls, after_text: str) -> bool:
        """Check if text following a directive contains a rationale."""
        return len(cls._find_after_separator(after_text)) >= cls.MIN_LENGTH

    @classmethod
    def extract(cls, after_text: str) -> str:
        """Extract the rationale text from the portion after a directive."""
        return cls._find_after_separator(after_text)

    @classmethod
    def is_vague(cls, rationale: str) -> bool:
        """Check if a rationale is too vague to be informative (SUP005)."""
        return bool(cls.VAGUE_RE.match(rationale.strip()))

    @classmethod
    def is_tautological(cls, rationale: str, codes: Sequence[str]) -> bool:
        """Check if a rationale merely restates the error description (SUP006)."""
        rationale_lower = rationale.strip().lower()
        for code in codes:
            for desc in cls.TAUTOLOGICAL.get(code.upper(), ()):
                if desc in rationale_lower:
                    return True
        return False

    @staticmethod
    def extract_noqa_codes(directive_text: str) -> Sequence[str]:
        """Extract error codes from a noqa directive string."""
        match = re.search(
            r'noqa\s*:\s*([A-Z0-9]+(?:\s*,\s*[A-Z0-9]+)*)',
            directive_text,
            re.IGNORECASE,
        )
        if match:
            return [code.strip().upper() for code in match.group(1).split(',')]
        return []


class SuppressionScanner:
    """Scans lines for suppression directives missing or having inadequate rationale.

    Uses a comment-first architecture: for each line, find where the comment
    starts (via ``_find_comment_start``), then search only within the comment
    portion. This makes false positives from directives inside string literals
    impossible by construction — no separate string-context check needed.
    """

    def __init__(self, filepath: Path, custom_prefixes: Sequence[str]) -> None:
        self.filepath = filepath
        self.custom_prefix_patterns: Sequence[tuple[str, re.Pattern[str]]] = [
            (prefix, build_custom_prefix_re(prefix)) for prefix in custom_prefixes
        ]
        self.violations: list[Violation] = []

    def check_line(self, lineno: int, line: str) -> None:
        """Check a single line for suppression directives without rationale."""
        comment_start = find_comment_start(line)
        if comment_start == -1:
            return

        comment = line[comment_start:]

        # Check our own directive (self-dogfooding)
        if DIRECTIVE_PREFIX.lower() in comment.lower():
            self._check_own_directive(lineno, line, comment)

        # Check standard directive types (noqa, type: ignore, pyright: ignore)
        for pattern, kind in STANDARD_CHECKS:
            self._check_directive(lineno, line, comment, pattern, kind)

        # Check custom linter directive prefixes
        self._check_custom_prefixes(lineno, line, comment)

    def _check_directive(
        self,
        lineno: int,
        line: str,
        comment: str,
        pattern: re.Pattern[str],
        kind: ViolationKind,
    ) -> None:
        """Check a directive pattern within the comment portion of a line."""
        match = pattern.search(comment)
        if match is None:
            return

        if is_in_backtick_span(comment, match.start()):
            return

        after = match.group('after')
        if not RationaleChecker.has_rationale(after):
            self._add_violation(lineno, line, kind, match.group())
            return

        # Rationale exists — check quality
        rationale = RationaleChecker.extract(after)
        if RationaleChecker.is_vague(rationale):
            self._add_violation(lineno, line, 'vague-rationale', match.group())
        elif kind == 'noqa-without-rationale':
            codes = RationaleChecker.extract_noqa_codes(match.group())
            if codes and RationaleChecker.is_tautological(rationale, codes):
                self._add_violation(lineno, line, 'tautological-rationale', match.group())

    def _check_own_directive(self, lineno: int, line: str, comment: str) -> None:
        """Check our own suppression directives for rationale.

        Violations are added directly to ``self.violations`` (bypassing
        ``_add_violation``) to avoid a circular self-suppression bypass where
        the self-directive check would find the directive on the same line and
        silently suppress the violation.
        """
        idx = comment.lower().find(DIRECTIVE_PREFIX.lower())
        if idx == -1:
            return

        after_prefix = comment[idx + len(DIRECTIVE_PREFIX) :]
        parts = after_prefix.strip()
        if not parts:
            return

        # skip-file is a structural directive, never requires rationale
        if 'skip-file' in parts.lower():
            return

        codes_match = re.match(
            r'[a-z0-9][-a-z0-9]*(?:\s*,\s*[a-z0-9][-a-z0-9]*)*',
            parts.strip(),
            re.IGNORECASE,
        )
        if codes_match:
            after_codes = parts[codes_match.end() :]
            if not RationaleChecker.has_rationale(after_codes):
                codes = [c.strip().lower() for c in codes_match.group().split(',')]
                for code in codes:
                    if code in ERROR_CODES:
                        # Add directly — bypass _add_violation to prevent circular suppression
                        self.violations.append(
                            Violation(
                                filepath=self.filepath,
                                line=lineno,
                                column=0,
                                kind=code,  # type: ignore[arg-type]  # code is validated against ERROR_CODES keys
                                source_line=line.rstrip().strip(),
                                directive_text=comment.strip(),
                            ),
                        )

    def _check_custom_prefixes(self, lineno: int, line: str, comment: str) -> None:
        """Check for custom linter suppression directives without rationale."""
        for _, pattern in self.custom_prefix_patterns:
            match = pattern.search(comment)
            if match is None:
                continue

            if is_in_backtick_span(comment, match.start()):
                continue

            # skip-file is a structural directive
            if 'skip-file' in comment[match.start() :].lower():
                continue

            after = match.group('after')
            if not RationaleChecker.has_rationale(after):
                self._add_violation(
                    lineno,
                    line,
                    'custom-suppress-without-rationale',
                    match.group(),
                )
            else:
                rationale = RationaleChecker.extract(after)
                if RationaleChecker.is_vague(rationale):
                    self._add_violation(lineno, line, 'vague-rationale', match.group())

    def _has_self_directive(self, line: str, kind: ViolationKind) -> bool:
        """Check if this line has our own suppression directive for the given kind."""
        lower = line.lower()
        prefix_lower = DIRECTIVE_PREFIX.lower()
        if prefix_lower not in lower:
            return False

        idx = lower.find(prefix_lower)
        codes_part = lower[idx + len(prefix_lower) :]

        # Strip trailing rationale after separator
        for sep in (' #', ' --', ' ---', ' \u2014', ' \u2013'):
            if sep in codes_part:
                codes_part = codes_part[: codes_part.find(sep)]

        codes = [c.strip().split()[0] for c in codes_part.split(',') if c.strip()]
        return kind in codes

    def _add_violation(self, lineno: int, line: str, kind: ViolationKind, directive_text: str = '') -> None:
        """Add a violation if not suppressed by our own directive."""
        if self._has_self_directive(line, kind):
            return

        stripped = line.rstrip()
        col = 0
        if directive_text:
            idx = line.find(directive_text.strip()[:10])
            if idx >= 0:
                col = idx

        self.violations.append(
            Violation(
                filepath=self.filepath,
                line=lineno,
                column=col,
                kind=kind,
                source_line=stripped.strip(),
                directive_text=directive_text.strip() if directive_text else '',
            ),
        )


# -- Utility Functions --------------------------------------------------------


def find_comment_start(line: str) -> int:
    """Find the position of the first ``#`` that starts a Python comment.

    Walks the line character-by-character, tracking string literal state.
    Handles single/double/triple quotes, string prefixes (r, b, f, u, and
    two-character combinations like rb, fr), and escape sequences.

    Returns -1 if no comment exists on this line.
    """
    string_delim: str | None = None
    i = 0

    while i < len(line):
        c = line[i]

        if string_delim is not None:
            # Inside a string literal
            if c == '\\':
                i += 2
                continue
            if len(string_delim) == 3 and line[i : i + 3] == string_delim:
                string_delim = None
                i += 3
                continue
            if len(string_delim) == 1 and c == string_delim:
                string_delim = None
                i += 1
                continue
            i += 1
        else:
            # Outside string literals
            if c == '#':
                return i
            if line[i : i + 3] in ('"""', "'''"):
                string_delim = line[i : i + 3]
                i += 3
                continue
            if c in ('"', "'"):
                string_delim = c
                i += 1
                continue
            # String prefixes (r, b, f, u and two-char combinations)
            if c in ('r', 'R', 'b', 'B', 'f', 'F', 'u', 'U') and i + 1 < len(line):
                rest = line[i + 1 :]
                # Two-char prefix (rb, br, rf, fr, etc.)
                if rest[0] in ('r', 'R', 'b', 'B', 'f', 'F'):
                    if len(rest) >= 4 and rest[1:4] in ('"""', "'''"):
                        string_delim = rest[1:4]
                        i += 5
                        continue
                    if len(rest) >= 2 and rest[1] in ('"', "'"):
                        string_delim = rest[1]
                        i += 3
                        continue
                # Single-char prefix
                if rest[0:3] in ('"""', "'''"):
                    string_delim = rest[0:3]
                    i += 4
                    continue
                if rest[0] in ('"', "'"):
                    string_delim = rest[0]
                    i += 2
                    continue
            i += 1

    return -1


def is_in_backtick_span(comment: str, match_start: int) -> bool:
    """Check if match position is inside a backtick-delimited code span.

    Catches documentation references like:
        ``# Use `# noqa: F841` to suppress warnings``
    """
    backtick_count = comment[:match_start].count('`')
    return backtick_count % 2 == 1


def track_multiline_strings(lines: Sequence[str]) -> Set[int]:
    """Return set of 1-indexed line numbers that are inside multi-line strings.

    Uses a simple state machine counting triple-quote delimiters.
    """
    in_multiline: set[int] = set()
    delimiter: str | None = None

    for i, line in enumerate(lines):
        lineno = i + 1
        stripped = line.strip()

        if delimiter is not None:
            in_multiline.add(lineno)
            if delimiter in stripped:
                count = stripped.count(delimiter)
                if count % 2 == 1:
                    delimiter = None
        else:
            for delim in ('"""', "'''"):
                count = stripped.count(delim)
                if count == 1:
                    delimiter = delim
                    break

    return in_multiline


# -- Output Formatting --------------------------------------------------------


def format_violation(v: Violation) -> str:
    """Format a violation for display."""
    code = ERROR_CODES[v.kind]
    message = VIOLATION_MESSAGES[v.kind]
    fix = FIX_SUGGESTIONS[v.kind]

    return (
        f'{v.filepath}:{v.line}:{v.column}: error: {code} {message}\n'
        f'    {v.source_line}\n'
        f'    Fix: {fix}\n'
        f'    Silence: {DIRECTIVE_PREFIX} {v.kind} — <explain why suppression is needed>'
    )


if __name__ == '__main__':
    sys.exit(main())
