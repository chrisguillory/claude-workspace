# replace_linter.py: skip-file
"""Structural edge case: model_copy(update=...) nested in a comprehension.

Expected: one RPL001 violation (ast.walk reaches nested calls).
"""
# ruff: noqa — test fixture, not real code

from __future__ import annotations

result = [hit.model_copy(update={'score': s}) for s, hit in scored]
