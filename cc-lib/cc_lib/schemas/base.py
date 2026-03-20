from __future__ import annotations

import os
from typing import Any

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
        serialize_by_alias=True,
    )


class CamelModel(StrictModel):
    """Strict model with automatic camelCase alias generation.

    For Claude Code's JSON protocol, which uses camelCase keys
    (hookSpecificOutput, permissionDecision, etc.) while Python convention
    is snake_case. Accepts both forms on input; serializes to camelCase
    with None fields excluded by default.
    """

    model_config = pydantic.ConfigDict(
        alias_generator=to_camel,
    )

    def model_dump(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Any,  # strict_typing_linter.py: loose-typing — override must match BaseModel.model_dump signature
    ) -> dict[str, Any]:  # strict_typing_linter.py: mutable-type — override must match BaseModel.model_dump signature
        return super().model_dump(exclude_none=exclude_none, **kwargs)

    def model_dump_json(
        self,
        *,
        exclude_none: bool = True,
        **kwargs: Any,  # strict_typing_linter.py: loose-typing — override must match BaseModel.model_dump_json signature
    ) -> str:
        return super().model_dump_json(exclude_none=exclude_none, **kwargs)
