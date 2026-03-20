# suppression_rationale_linter.py: skip-file
# strict_typing_linter.py: skip-file
# exception_safety_linter.py: skip-file
# ruff: noqa
"""Instructive test cases for suppression_rationale_linter.py.

Each test is a pair: a tag line (# EXPECT: SUPxxx or # OK) followed by the
line being tested. The test runner associates each tag with the next line.

This file is excluded from ruff and from this linter's own skip-file check
(via --no-skip-file in the test runner).
"""

from __future__ import annotations


# -- SUP001: noqa-without-rationale -------------------------------------------

# EXPECT: SUP001
x = 1  # noqa: F841
# EXPECT: SUP001
x = 1  # noqa: F841, E501
# EXPECT: SUP001
x = 1  # noqa

# -- SUP001: noqa with valid rationale (no violation) -------------------------

# OK
x = 1  # noqa: F841 — assigned for side effect in test setup
# OK
x = 1  # noqa: F841 -- assigned for side effect
# OK
x = 1  # noqa: F841 --- assigned for side effect
# OK
x = 1  # noqa: F841 # assigned for side effect
# OK
x = 1  # noqa: F841 - assigned for side effect

# -- SUP002: type-ignore-without-rationale ------------------------------------

# EXPECT: SUP002
x: int = 'hello'  # type: ignore[assignment]
# EXPECT: SUP002
x: int = 'hello'  # type: ignore

# -- SUP002: type-ignore with valid rationale (no violation) ------------------

# OK
x: int = 'hello'  # type: ignore[assignment] — json.loads returns Any
# OK
x: int = 'hello'  # type: ignore[assignment] -- json.loads returns Any

# -- SUP003: pyright-ignore-without-rationale ---------------------------------

# EXPECT: SUP003
x = 1  # pyright: ignore[reportGeneralClassIssues]
# EXPECT: SUP003
x = 1  # pyright: ignore

# -- SUP003: pyright-ignore with valid rationale (no violation) ---------------

# OK
x = 1  # pyright: ignore[reportGeneralClassIssues] — stub mismatch

# -- SUP004: custom-suppress-without-rationale --------------------------------

# EXPECT: SUP004
x = [1, 2, 3]  # strict_typing_linter.py: mutable-type
# EXPECT: SUP004
x = 1  # exception_safety_linter.py: bare-except

# -- SUP004: custom-suppress with valid rationale (no violation) --------------

# OK
x = [1, 2, 3]  # strict_typing_linter.py: mutable-type — pydantic requires list

# -- SUP004: skip-file is structural, never requires rationale ----------------

# OK
# strict_typing_linter.py: skip-file
# OK
# exception_safety_linter.py: skip-file

# -- SUP005: vague-rationale --------------------------------------------------

# EXPECT: SUP005
x = 1  # noqa: F841 — intentional
# EXPECT: SUP005
x = 1  # noqa: F841 -- needed
# EXPECT: SUP005
x = 1  # noqa: F841 — safe here
# EXPECT: SUP005
x = 1  # noqa: F841 -- ok
# EXPECT: SUP005
x = 1  # noqa: F841 — because
# EXPECT: SUP005
x = 1  # noqa: F841 -- fine
# EXPECT: SUP005
x = 1  # noqa: F841 — n/a
# EXPECT: SUP005
x = 1  # noqa: F841 -- todo
# EXPECT: SUP005
x = 1  # noqa: F841 — this is intentional
# EXPECT: SUP005
x = 1  # noqa: F841 -- suppress this
# EXPECT: SUP005
x = 1  # type: ignore[override] — needed

# -- SUP005: specific enough rationale (no violation) -------------------------

# OK
x = 1  # noqa: F841 — needed for pytest parametrize fixture
# OK
x = 1  # noqa: F841 -- intentional side effect in test setup
# OK
x = 1  # noqa: F841 — safe because return type is narrowed by isinstance

# -- SUP006: tautological-rationale -------------------------------------------

# EXPECT: SUP006
x = 1  # noqa: F841 — unused variable
# EXPECT: SUP006
x = 1  # noqa: F401 -- unused import
# EXPECT: SUP006
x = 1  # noqa: E501 — line too long
# EXPECT: SUP006
x = 1  # noqa: E722 -- bare except
# EXPECT: SUP006
x = 1  # noqa: B006 — mutable default
