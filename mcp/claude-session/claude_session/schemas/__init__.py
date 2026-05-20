"""Schema definitions for claude-session.

This package contains Pydantic models for various data schemas:
- session: Claude Code session JSONL record types
- operations: Service operation result schemas (archive, restore, delete, lineage)
- cc_internal_api: Wire schemas Claude Code sends/receives over HTTP
  (Anthropic ``/v1/messages``, ``/api/event_logging/batch``, Statsig ``/v1/initialize``,
  feature-flag ``/api/eval/sdk-*``, etc.) — captured via mitmproxy
- captures: Other captured third-party wire schemas (Datadog, GCS, Segment)

Wire-schema version fields stay ``str``, not ``cc_lib.types.CCVersion``. The
schema's job is to faithfully model what's observed; CCVersion-typing risks
rejecting captures on unilateral wire-format drift (RC suffixes, dev markers,
future formats) and lossy normalization on serialization. Applies to: session
JSONL record ``version`` fields, every ``version``-like field under
``cc_internal_api/``, and version-bearing bodies under ``captures/``.
"""

from __future__ import annotations

from claude_session.schemas.base import StrictModel
from claude_session.schemas.types import JsonDatetime, ModelId

__all__ = [
    'JsonDatetime',
    'ModelId',
    'StrictModel',
]
