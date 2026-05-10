"""Wire-format type aliases shared between the HTTP bridge and its CLI client.

These aliases describe the HTTP contract between ``bridge.py`` (FastAPI server)
and ``cli/main.py`` (httpx client). Living here lets the CLI import them without
pulling in FastAPI or BrowserService through ``bridge.py``.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    'OnErrorPolicy',
    'PipelineStatus',
    'StepStatus',
    'ToolStatus',
]

type ToolStatus = Literal['ok', 'error']
type StepStatus = Literal['ok', 'error', 'skipped']
type PipelineStatus = Literal['completed', 'partial']
type OnErrorPolicy = Literal['stop', 'continue']
