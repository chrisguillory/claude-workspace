# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""trailing-comma USED: __all__ lacks a trailing comma; the directive suppresses it."""

__all__ = ['a', 'b']  # strict_typing_linter.py: trailing-comma

a = 1
b = 2
