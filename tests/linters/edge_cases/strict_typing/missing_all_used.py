# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""missing-all USED: no __all__, so the directive on the first definition suppresses it."""


def thing() -> None:  # strict_typing_linter.py: missing-all
    pass
