"""Pydantic base models for three trust levels of extra field handling.

Terminology follows JSON Schema: "closed" rejects additional properties,
"open" accepts them. Trust level determines the default:

- ClosedModel: internal data we construct. extra='forbid' always.
- StrictModel: protocol data (Claude Code, MCP). extra='allow' by default.
- OpenModel: external data (Chrome, CDP). extra='allow' by default.
- CamelModel: StrictModel with camelCase JSON serialization.
"""

from __future__ import annotations

import os
from typing import Any

import pydantic
from pydantic.alias_generators import to_camel


class ClosedModel(pydantic.BaseModel):
    """Closed content model for internal data.

    Use when both producer and consumer are our code. Unknown fields
    indicate a programming bug and are always rejected.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid',
        frozen=True,
        strict=True,
        validate_default=True,
        use_attribute_docstrings=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class StrictModel(pydantic.BaseModel):
    """Protocol data model — forward-compatible with upstream schema evolution.

    Unknown fields are preserved in ``model_extra`` so upstream additions
    (Claude Code adding new hook fields) don't break consumers.

    Set ``CC_STRICT_MODEL_EXTRA_FORBID=1`` to surface new fields as
    validation errors — for developers tracking upstream protocol
    changes who want to keep their schemas current.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid' if os.environ.get('CC_STRICT_MODEL_EXTRA_FORBID') == '1' else 'allow',
        frozen=True,
        strict=True,
        validate_default=True,
        use_attribute_docstrings=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )


class CamelModel(StrictModel):
    """Protocol model with automatic camelCase alias generation.

    For Claude Code's JSON protocol, which uses camelCase keys
    while Python convention is snake_case. Accepts both forms on input;
    serializes to camelCase with None fields excluded by default.
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


class OpenModel(pydantic.BaseModel):
    """Open content model for external data with no stable schema.

    Use for data from systems we don't control (Chrome Local State,
    CDP, IndexedDB) where the upstream schema evolves independently.

    Set ``CC_OPEN_MODEL_EXTRA_FORBID=1`` to reject unknown fields —
    useful for gradual typing: discover schema drift, add new fields
    to the model, then switch back to allow.
    """

    model_config = pydantic.ConfigDict(
        extra='forbid' if os.environ.get('CC_OPEN_MODEL_EXTRA_FORBID') == '1' else 'allow',
        frozen=True,
        strict=True,
        validate_default=True,
        use_attribute_docstrings=True,
        validate_by_alias=True,
        validate_by_name=True,
        serialize_by_alias=True,
    )
