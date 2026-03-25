# reexport_linter.py: skip-file
"""Structural edge case: __all__ contains only local definitions.

Expected: zero violations.
"""
# ruff: noqa — test fixture, not real code

from os.path import join  # imported but NOT in __all__


def helper() -> None:
    """A locally defined function."""


TIMEOUT = 30

__all__ = [
    'helper',
    'TIMEOUT',
]
