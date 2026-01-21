"""Shared type aliases for Claude Code workspace."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import pydantic

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']
type SessionEndReason = Literal['prompt_input_exit', 'clear', 'logout', 'other']

# Pydantic-enhanced types for JSON serialization (allows string→type conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
JsonUuid = Annotated[UUID, pydantic.Field(strict=False)]
