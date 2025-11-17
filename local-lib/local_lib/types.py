"""Shared type aliases for Claude Code workspace."""

from datetime import datetime
from typing import Annotated, Literal
import pydantic

type SessionState = Literal["active", "exited", "completed", "crashed"]
type SessionSource = Literal["startup", "resume", "compact", "clear"]
type SessionEndReason = Literal["prompt_input_exit", "clear", "logout", "other"]

# Pydantic-enhanced datetime for JSON serialization (allows string→datetime conversion)
JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
