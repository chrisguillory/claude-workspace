# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""ordering USED: a private definition before a public one; the swap double-emits, so both carry a directive."""

__all__ = []


def _helper() -> None:  # strict_typing_linter.py: ordering
    pass


def public_api() -> None:  # strict_typing_linter.py: ordering
    pass
