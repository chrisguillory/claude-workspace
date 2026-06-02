from __future__ import annotations

__all__ = [
    'Meeting',
]

from cc_lib.schemas.base import ClosedModel


class Meeting(ClosedModel):
    """A meeting in summary form — granola-kit's projected list result."""

    id: str
    title: str | None
    created_at: str
    is_shared: bool
    type: str | None
