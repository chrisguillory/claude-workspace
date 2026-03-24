# reexport_linter.py: skip-file
"""Test fixture for reexport_linter.py — instructive test cases.

Each function-scoped section demonstrates one pattern. Violation cases are
grouped first, then correct (non-flagged) cases.

NOTE: This file uses module-level structure (imports and __all__) rather
than function-scoped patterns because the linter operates on module-level
constructs.  The file itself acts as the test fixture — run the linter
against it and check which names in __all__ are flagged.
"""
# ruff: noqa — test fixture, not real code

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports that ARE re-exports (should flag REX001)
# ---------------------------------------------------------------------------

from os.path import join  # imported from os.path
from collections import OrderedDict  # imported from collections
import json  # module import


# ---------------------------------------------------------------------------
# Local definitions (should NOT flag)
# ---------------------------------------------------------------------------


def local_function() -> None:
    """A locally defined function."""


class LocalClass:
    """A locally defined class."""


LOCAL_CONSTANT = 42


# ---------------------------------------------------------------------------
# Mixed: import + local definition with same name = shadow (NOT flagged)
# The local definition takes precedence.
# ---------------------------------------------------------------------------

from typing import Any  # noqa: F811 — intentional shadow for test

Any = object  # noqa: F811 — shadows the import, local def wins


# ---------------------------------------------------------------------------
# __all__ listing both re-exports and local definitions
# ---------------------------------------------------------------------------

__all__ = [
    # Re-exports (should flag REX001)
    'join',
    'OrderedDict',
    'json',
    # Local definitions (should NOT flag)
    'local_function',
    'LocalClass',
    'LOCAL_CONSTANT',
    # Shadowed (should NOT flag — local def exists)
    'Any',
]
