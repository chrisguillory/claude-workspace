"""Strict Pydantic base model."""

from __future__ import annotations

import pydantic

__all__ = [
    'StrictModel',
]


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation.

    Config:
    - extra='forbid': Reject unknown fields (fail-fast)
    - strict=True: No implicit type coercion
    - frozen=True: Immutable after creation
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',
        strict=True,
        frozen=True,
    )
