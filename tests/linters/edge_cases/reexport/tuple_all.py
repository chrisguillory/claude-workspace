# reexport_linter.py: skip-file
"""Structural edge case: __all__ = ('a', 'b') (tuple form).

Expected: same behavior as list form.
Tuple parsed via ast.Tuple, should produce violations.
"""
# ruff: noqa — test fixture, not real code

from os.path import join
from collections import OrderedDict


def local_func() -> None:
    """A locally defined function."""


__all__ = (
    'join',  # re-export — SHOULD flag
    'OrderedDict',  # re-export — SHOULD flag
    'local_func',  # local def — should NOT flag
)
