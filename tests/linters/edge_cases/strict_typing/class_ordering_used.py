# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""class-ordering USED: a function before a class; check_module_ordering double-emits, so both carry a directive."""

__all__ = []


def func() -> None:  # strict_typing_linter.py: class-ordering
    pass


class Klass:  # strict_typing_linter.py: class-ordering
    pass
