# reexport_linter.py: skip-file
"""Structural edge case: __all__: list[str] = [...] (annotated assignment).

Expected: same behavior as plain __all__ = [...].
Annotated form parsed via ast.AnnAssign, should produce violations.
"""
# ruff: noqa — test fixture, not real code

from os.path import join
from collections import OrderedDict


def local_func() -> None:
    """A locally defined function."""


__all__: list[str] = [
    'join',  # re-export — SHOULD flag
    'OrderedDict',  # re-export — SHOULD flag
    'local_func',  # local def — should NOT flag
]
