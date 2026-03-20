# suppression_rationale_linter.py: skip-file
# strict_typing_linter.py: skip-file
# exception_safety_linter.py: skip-file
# ruff: noqa
"""Edge cases and regression tests for suppression_rationale_linter.py.

Tests false positive prevention, string context handling, multi-line strings,
backtick spans, and boundary conditions. Tag lines (# EXPECT / # OK) are
on the line ABOVE the line being tested.
"""

from __future__ import annotations

# -- String context: directives inside strings should NOT trigger -------------

# OK
x = '# noqa: F841'
# OK
x = '# type: ignore[assignment]'
# OK
x = f'value: {1}  # noqa: E501'
# OK
x = r'# noqa: F841'
# OK
x = b'# noqa: F841'

# -- String + real directive on same line -------------------------------------

# EXPECT: SUP001
x = 'has # noqa inside'  # noqa: F841

# -- Backtick spans: directives in documentation should NOT trigger -----------

# OK
# Use `# noqa: F841` to suppress unused variable warnings
# OK
# The `# type: ignore[override]` directive overrides the check

# -- Multi-line strings: directives inside should NOT trigger -----------------

docstring = """
This docstring mentions # noqa: F841 but it's inside a string.
And also # type: ignore[assignment] which should be ignored.
"""

raw_triple = """
# noqa: E501
# type: ignore
"""

# -- Minimum rationale length boundary ---------------------------------------

# OK
x = 1  # noqa: F841 — xy
# EXPECT: SUP001
x = 1  # noqa: F841 —
# EXPECT: SUP001
x = 1  # noqa: F841 --

# -- Case insensitivity -------------------------------------------------------

# EXPECT: SUP001
x = 1  # NOQA: F841
# EXPECT: SUP002
x = 1  # Type: Ignore[assignment]
# EXPECT: SUP003
x = 1  # Pyright: Ignore

# -- SUP005 vague with different directive types ------------------------------

# EXPECT: SUP005
x = 1  # type: ignore[override] — ok
# EXPECT: SUP005
x = 1  # pyright: ignore[reportMissing] — fine

# -- SUP006 tautological with less common codes ------------------------------

# EXPECT: SUP006
x = 1  # noqa: PLC0415 — import outside top level
# EXPECT: SUP006
x = 1  # noqa: S101 — use of assert

# -- SUP006 partial match should still catch ----------------------------------

# EXPECT: SUP006
x = 1  # noqa: F841 — the variable is unused here

# -- Lines with no comment at all (no violation) ------------------------------

x = 1
y = x + 2
z = [1, 2, 3]

# -- Pure comment lines with no suppression directive (no violation) ----------

# This is a regular comment
# TODO: fix this later
# FIXME: edge case not handled

# -- Unicode em dash (U+2014) and en dash (U+2013) as separators -------------

# OK
x = 1  # noqa: F841 — assigned for test fixture setup
# OK
x = 1  # noqa: F841 – assigned for test fixture setup
