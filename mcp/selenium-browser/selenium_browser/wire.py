"""Wire-format types shared between the HTTP bridge and its CLI client.

These describe the HTTP contract between ``bridge.py`` (FastAPI server)
and ``cli/main.py`` (httpx client). Living here lets the CLI import them without
pulling in FastAPI or BrowserService through ``bridge.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from cc_lib.schemas.base import ClosedModel

__all__ = [
    'OnErrorPolicy',
    'PipelineStatus',
    'PipelineStep',
    'StepStatus',
    'ToolStatus',
]

type ToolStatus = Literal['ok', 'error']
type StepStatus = Literal['ok', 'error', 'skipped']
type PipelineStatus = Literal['completed', 'partial']
type OnErrorPolicy = Literal['stop', 'continue']


class PipelineStep(ClosedModel):
    """One step in a pipeline batch."""

    tool: str
    params: Mapping[str, Any] = {}  # strict_typing_linter.py: loose-typing — validated at dispatch
