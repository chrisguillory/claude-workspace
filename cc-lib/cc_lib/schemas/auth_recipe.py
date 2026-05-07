from __future__ import annotations

__all__ = [
    'AuthRecipe',
    'AuthRecipeStep',
]

from collections.abc import Sequence

from cc_lib.schemas.base import ClosedModel
from cc_lib.types import JsonObject


class AuthRecipe(ClosedModel):
    """A versioned auth pipeline: ordered steps plus declared placeholders.

    String values inside ``steps[].input`` matching ``{{key}}`` are dynamic
    fields whose names are listed in ``placeholders``.
    """

    schema_version: str = '1.0'
    name: str
    description: str
    placeholders: Sequence[str]
    steps: Sequence[AuthRecipeStep]


class AuthRecipeStep(ClosedModel):
    """One step: a tool ``name`` and its ``input`` payload."""

    name: str
    input: JsonObject
