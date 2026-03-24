# reexport_linter.py: skip-file
"""Edge case fixture for reexport_linter.py — regression tests.

Tests boundary conditions and false positive prevention.
Each section is annotated with expected linter behavior.
"""
# ruff: noqa — test fixture, not real code

from __future__ import annotations

from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Edge: TYPE_CHECKING import (SHOULD flag — re-export is re-export)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Edge: Aliased import (from x import Y as Z — Z is checked)
# ---------------------------------------------------------------------------

from os.path import exists as path_exists  # aliased import


# ---------------------------------------------------------------------------
# Edge: Module import (import os — flagged if in __all__)
# ---------------------------------------------------------------------------

import os  # module import


# ---------------------------------------------------------------------------
# Edge: type alias (type Foo = ... is a local definition, NOT a re-export)
# ---------------------------------------------------------------------------

type StringAlias = str  # PEP 695 type alias — local definition


# ---------------------------------------------------------------------------
# Edge: Annotated assignment (local definition, NOT a re-export)
# ---------------------------------------------------------------------------

SOME_VALUE: int = 99


# ---------------------------------------------------------------------------
# Edge: Relative import (from . import X — SHOULD flag)
# ---------------------------------------------------------------------------

from . import some_submodule


# ---------------------------------------------------------------------------
# __all__ for edge case testing
# ---------------------------------------------------------------------------

__all__ = [
    # TYPE_CHECKING import — SHOULD flag REX001 (re-export is re-export)
    'Path',
    # Aliased import — SHOULD flag REX001 (Z is the re-exported name)
    'path_exists',
    # Module import — SHOULD flag REX001
    'os',
    # type alias — should NOT flag (local definition)
    'StringAlias',
    # Annotated assignment — should NOT flag (local definition)
    'SOME_VALUE',
    # Relative import — SHOULD flag REX001
    'some_submodule',
]
