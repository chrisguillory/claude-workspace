# reexport_linter.py: skip-file
"""Structural edge case: star import causes file to be skipped.

Expected: zero violations (file skipped entirely due to unknowable names).
"""
# ruff: noqa — test fixture, not real code

from os import *  # noqa: F403 — star import for test

__all__ = [
    'path',  # would be a re-export, but file is skipped
]
