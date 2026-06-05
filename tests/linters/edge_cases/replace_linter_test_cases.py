# replace_linter.py is the linter under test; the others skip this file.
# exception_safety_linter.py: skip-file
# strict_typing_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""Instructive fixture for replace_linter: which model_copy forms flag RPL001.

Each call is tagged ``FLAG`` (must trigger RPL001) or ``CLEAN`` (must not). The
boundary is the ``update=`` keyword: bare clones and ``deep=True``-only clones are
plain copies that ``__replace__`` does not replace, so they stay clean; any call
carrying ``update=`` — literal, dynamic, or alongside ``deep=`` — is flagged.

The ``model`` placeholder is an undefined name; the linter is syntactic and never
resolves the receiver type, so the fixture needs no real Pydantic model.
"""

from __future__ import annotations

some_update = {'field': 1}


def flag_literal_update() -> object:
    """FLAG: update= with a dict literal — the canonical case."""
    return model.model_copy(update={'field': 1})


def flag_dynamic_update() -> object:
    """FLAG: update= bound to a name — still typed dict[str, Any], still flagged."""
    return model.model_copy(update=some_update)


def flag_deep_and_update() -> object:
    """FLAG: update= alongside deep=True — the update keyword is what matters."""
    return model.model_copy(deep=True, update={'field': 1})


def flag_multiline_update() -> object:
    """FLAG: update= spanning multiple lines below the model_copy token."""
    return model.model_copy(
        update={'field': 1},
    )


def flag_in_comprehension() -> list[object]:
    """FLAG: update= inside a comprehension — ast.walk reaches nested calls."""
    return [item.model_copy(update={'field': 1}) for item in items]


def clean_bare_copy() -> object:
    """CLEAN: no update= — a plain clone __replace__ does not replace."""
    return model.model_copy()


def clean_deep_copy() -> object:
    """CLEAN: deep=True without update= — still a plain clone."""
    return model.model_copy(deep=True)


def clean_model_dump() -> object:
    """CLEAN: model_dump, not model_copy — different method name."""
    return model.model_dump(update={'field': 1})


def clean_other_update() -> object:
    """CLEAN: a plain .update() call — not model_copy."""
    return obj.update({'field': 1})


def clean_suppressed_update() -> object:
    """CLEAN: a real RPL001 silenced by an inline directive."""
    return model.model_copy(update={'field': 1})  # replace_linter.py: model-copy-update
