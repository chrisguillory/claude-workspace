# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""class-ordering UNUSED: class before function is correct, so the directive matches nothing."""

__all__ = []


class Klass:
    pass


def func() -> None:  # strict_typing_linter.py: class-ordering
    pass
