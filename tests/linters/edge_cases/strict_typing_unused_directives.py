# ruff: noqa: E402
# exception_safety_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
"""Test cases for --report-unused-directives in strict_typing_linter.py.

Each case has a comment indicating the expected behavior:
- STALE: directive should be flagged as unused
- ACTIVE: directive suppresses a real violation, should NOT be flagged
"""

from __future__ import annotations

from collections.abc import Sequence

# -- STALE: mutable-type on an immutable annotation ---------------------------

x: Sequence[int] = []  # strict_typing_linter.py: mutable-type — not actually mutable

# -- ACTIVE: mutable-type on an actual mutable type --------------------------

y: list = []  # strict_typing_linter.py: mutable-type — intentional mutable interface

# -- STALE: loose-typing on a concrete type -----------------------------------

z: int = 0  # strict_typing_linter.py: loose-typing — not actually loose

# -- ACTIVE: loose-typing on Any ---------------------------------------------

from typing import Any

w: Any = None  # strict_typing_linter.py: loose-typing — dynamic payload
