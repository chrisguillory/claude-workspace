# reexport_linter.py: skip-file
"""Structural edge case: empty __all__ = [].

Expected: zero violations (not an error).
"""
# ruff: noqa — test fixture, not real code

from os.path import join  # imported but not in __all__

__all__: list[str] = []
