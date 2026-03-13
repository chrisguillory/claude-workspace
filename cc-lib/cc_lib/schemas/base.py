from __future__ import annotations

import os

import pydantic
from pydantic.alias_generators import to_camel


class StrictModel(pydantic.BaseModel):
    """Base model: strict types, immutable, forward-compatible by default.

    Unknown fields are preserved (``extra='allow'``) so upstream schema
    evolution doesn't break consumers. Preserved extras are accessible via
    ``__pydantic_extra__`` for drift detection.

    Set ``CC_SCHEMA_EXTRA_FORBID=1`` to switch to ``extra='forbid'``.

    Use ``__replace__()`` (Python 3.13+, PEP 681) for type-safe mutations
    instead of ``model_copy(update={...})`` which is untyped.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid' if os.environ.get('CC_SCHEMA_EXTRA_FORBID') == '1' else 'allow',
        frozen=True,
        strict=True,
        validate_default=True,
        use_attribute_docstrings=True,
        validate_by_alias=True,
        validate_by_name=True,
    )


class CamelModel(StrictModel):
    """Strict model with automatic camelCase alias generation.

    For Claude Code's JSON protocol, which uses camelCase keys
    (hookSpecificOutput, permissionDecision, etc.) while Python convention
    is snake_case. Accepts both forms on input; serializes to camelCase.

    Serialize with: ``model_dump_json(by_alias=True, exclude_none=True)``
    """

    model_config = pydantic.ConfigDict(
        alias_generator=to_camel,
    )
