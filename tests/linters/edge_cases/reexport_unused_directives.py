# reexport_linter.py is the linter under test; the others skip this file.
# exception_safety_linter.py: skip-file
# strict_typing_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa
"""``used_``/``unused_`` fixture for reexport_linter --report-unused-directives.

REX001 (reexported-symbol) fires on an imported name listed in ``__all__``. The directive on
the imported ``used_reexport`` entry suppresses a real REX001; the directive on the local
``unused_reexport`` entry matches nothing. The two are kept well apart in ``__all__`` so the
unused directive stays outside the 4-line backward proximity window of ``used_reexport``'s
violation — otherwise it back-matches and is silently treated as used (the reexport
``__all__``-adjacency trap).

Directives are line-keyed (they sit on ``__all__`` entries, not in def/class ranges), so the
test locates them by entry name rather than via ``sole_directive_line``.
"""

from __future__ import annotations

from collections.abc import Mapping as used_reexport


def unused_reexport() -> None:
    """A local definition; listing it in __all__ is not a re-export, so its directive is unused."""


__all__ = [
    'used_reexport',  # reexport_linter.py: reexported-symbol
    # Spacer: keep the unused directive below >=5 lines under 'used_reexport' (the only
    # REX001 in this file), clear of its [D-4, D] backward proximity window — otherwise it
    # back-matches that violation and is silently treated as used.
    #
    #
    'unused_reexport',  # reexport_linter.py: reexported-symbol
]
