# replace_linter.py is the linter under test; the others skip this file.
# exception_safety_linter.py: skip-file
# strict_typing_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""``used_``/``unused_`` fixture for replace_linter --report-unused-directives.

RPL001 (model-copy-update) fires on a ``model_copy(update=...)`` call. The directive
on ``used_model_copy_update``'s call suppresses a real RPL001; the directive in
``unused_model_copy_update`` sits on a bare ``model_copy()`` (no update=, so no
violation) and matches nothing.

The two functions are kept apart so the unused directive stays outside the [D-4, D]
backward proximity window of the used violation — otherwise it back-matches and is
silently treated as used.
"""

from __future__ import annotations


def used_model_copy_update() -> object:
    """A real RPL001, silenced — the directive is used."""
    return model.model_copy(update={'field': 1})  # replace_linter.py: model-copy-update


def spacer_keeps_directives_apart() -> None:
    """Spacer so the unused directive is >=5 lines below the used violation."""


def unused_model_copy_update() -> object:
    """A bare clone carries no RPL001, so this directive matches nothing."""
    return model.model_copy()  # replace_linter.py: model-copy-update
