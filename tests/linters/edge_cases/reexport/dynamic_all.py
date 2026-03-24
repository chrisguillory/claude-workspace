# reexport_linter.py: skip-file
"""Structural edge case: __all__ += ['extra'] after static __all__.

Expected: zero violations (file skipped due to dynamic __all__).
The augmented assignment makes the full contents unknowable statically.
"""
# ruff: noqa — test fixture, not real code

from os.path import join

__all__ = ['join']
__all__ += ['extra']  # dynamic mutation — linter bails out
