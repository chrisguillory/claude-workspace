# exception_safety_linter.py: skip-file
# reexport_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""mutable-type matched by PROXIMITY (Strategy 2), locking in find_unused_directives' ``| {code}`` union.

Reproduces claude_session/.../loader.py:246-247: a multi-line signature with two ``dict`` violations
(param + return) and a directive on each. During checking, ``_has_directive`` scans the signature scope
and records only the *first* directive it finds (the param one) — for both violations — so the *return*
directive is never recorded and find_unused_directives must fall back to the 4-line backward window
(Strategy 2), unlike the single-line field fixtures (Strategy 1). The raw violation is stored as the
code ``mutable-type``; the matcher must union ``{code}`` because ``CODE_TO_KINDS['mutable-type']`` is
``{'mutable'}`` — which alone would miss it and falsely flag the return directive unused. Both directives
being USED (not flagged) is the regression guard for the union.
"""

__all__ = []


def proximity(
    data: dict[str, int],
    flag: bool = False,  # strict_typing_linter.py: mutable-type
) -> dict[str, int]:  # strict_typing_linter.py: mutable-type
    return data
