# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""trailing-comma UNUSED: __all__ has a trailing comma, so the directive matches nothing."""

__all__ = [  # strict_typing_linter.py: trailing-comma
    'a',
    'b',
]

a = 1
b = 2
