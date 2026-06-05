# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""ordering UNUSED: public before private is correct, so the directive matches nothing."""

__all__ = []


def public_api() -> None:
    pass


def _helper() -> None:  # strict_typing_linter.py: ordering
    pass
