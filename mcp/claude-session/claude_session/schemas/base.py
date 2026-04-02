"""
Shared Pydantic base model for strict validation.

All Pydantic schema models in the application should inherit from StrictModel.
This module re-exports BaseStrictModel as StrictModel for the operations/ package.
"""

from __future__ import annotations

from claude_session.schemas.types import BaseStrictModel

__all__ = [
    'StrictModel',
]


class StrictModel(BaseStrictModel):
    """Operations-layer strict model.

    Inherits from BaseStrictModel (extra='forbid', strict=True, frozen=True).
    Used by claude_session/schemas/operations/ package.
    """

    pass
