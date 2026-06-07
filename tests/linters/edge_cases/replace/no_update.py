# replace_linter.py: skip-file
"""Structural edge case: model_copy without update= is never flagged.

Expected: zero violations (bare and deep-only clones are plain copies).
"""
# ruff: noqa — test fixture, not real code

from __future__ import annotations

a = model.model_copy()
b = model.model_copy(deep=True)
c = model.model_dump(update={'field': 1})  # different method name
