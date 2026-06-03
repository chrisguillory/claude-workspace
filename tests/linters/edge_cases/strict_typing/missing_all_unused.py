# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""missing-all UNUSED: __all__ is defined, so the missing-all directive matches nothing."""

__all__ = [
    'thing',
]


def thing() -> None:  # strict_typing_linter.py: missing-all
    pass
