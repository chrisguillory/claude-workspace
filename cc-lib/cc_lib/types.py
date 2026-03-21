"""Shared type aliases for Claude Code workspace."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import pydantic

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']

# Pydantic-enhanced types for JSON serialization (allows string→type conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
JsonUuid = Annotated[UUID, pydantic.Field(strict=False)]

# External JSON object with no published schema — intentional Any, not lazy typing.
# Prefer specific Pydantic models when the structure is known.
type JsonObject = Mapping[str, Any]
