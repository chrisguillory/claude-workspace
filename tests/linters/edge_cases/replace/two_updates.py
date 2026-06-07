# replace_linter.py: skip-file
"""Structural edge case: two model_copy(update=...) calls in one file.

Expected: two RPL001 violations.
"""
# ruff: noqa — test fixture, not real code

from __future__ import annotations

a = first.model_copy(update={'x': 1})
b = second.model_copy(deep=True, update={'y': 2})
