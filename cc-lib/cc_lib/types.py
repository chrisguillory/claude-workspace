"""Shared type aliases for Claude Code workspace."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import pydantic
from pydantic import JsonValue

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']

# Pydantic-enhanced types for JSON serialization (allows string→type conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
JsonUuid = Annotated[UUID, pydantic.Field(strict=False)]

# JSON object types — use when the structure has no published schema.
# Prefer specific Pydantic models when the structure is known.
#
# JsonObject: opaque — skips value validation. Use for data already parsed
# from JSON (Chrome Local State, CDP responses) where re-validation is redundant.
#
# StrictJsonObject: validated — rejects non-JSON values (set, bytes, datetime)
# via pydantic.JsonValue. Use when constructing JSON to send or when data
# provenance is uncertain.
type JsonObject = Mapping[str, Any]
type StrictJsonObject = Mapping[str, JsonValue]
