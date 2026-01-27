"""Claude Code hook input schemas.

See: https://code.claude.com/docs/en/hooks
"""

from __future__ import annotations

from typing import Literal

import pydantic


class StrictModel(pydantic.BaseModel):
    """Base model with strict validation."""

    model_config = pydantic.ConfigDict(
        extra='forbid',  # Reject unknown fields (fail-fast)
        strict=True,  # Strict type coercion
        frozen=True,  # Immutable after creation
    )


class SessionStartHookInput(StrictModel):
    """SessionStart hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionstart
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['SessionStart']
    source: Literal['startup', 'resume', 'compact', 'clear']
    model: str | None = None


class SessionEndHookInput(StrictModel):
    """SessionEnd hook input schema.

    See: https://code.claude.com/docs/en/hooks#sessionend
    """

    session_id: str
    cwd: str
    transcript_path: str
    hook_event_name: Literal['SessionEnd']
    reason: Literal['prompt_input_exit', 'clear', 'logout', 'other']
