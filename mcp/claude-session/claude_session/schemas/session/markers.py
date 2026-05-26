"""Field markers for special serialization handling.

PathMarker enables runtime discovery of filesystem path fields for cross-machine
session transfer. CCVersionMarker brands fields that carry a Claude Code
version string (wire type stays ``str`` per the wire-schema retention policy in
``schemas/__init__.py``; the marker only adds type-system provenance).
Uses Annotated pattern (Pydantic v2 recommended approach).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from claude_session.schemas.types import PathStr

__all__ = [
    'CCVersionMarker',
    'CCVersionStrField',
    'PathField',
    'PathListField',
    'PathMarker',
]


@dataclass(frozen=True)
class PathMarker:
    """Mark field as filesystem path for translation."""


type PathField = Annotated[PathStr, PathMarker()]
"""A path that needs translation during cross-machine restore."""

type PathListField = Annotated[list[PathStr], PathMarker()]
"""A list of paths that need translation during cross-machine restore."""


@dataclass(frozen=True)
class CCVersionMarker:
    """Mark field as a Claude Code version string.

    Distinct from ``CCVersion`` (the parsed ``packaging.Version`` subclass): the
    marker brands wire-schema ``str`` fields whose runtime type must stay
    primitive per the wire-schema retention policy in ``schemas/__init__.py``.
    Consumers parse to ``CCVersion`` at read time.

    >>> # noinspection PyUnresolvedReferences
    >>> from cc_lib.types import CCVersion
    """


type CCVersionStrField = Annotated[str, CCVersionMarker()]
"""A Claude Code version string in wire data (PEP 440 shape, stays ``str``)."""
