"""Shared type aliases for Claude Code workspace.

JSON value types — three levels for external data with no published schema:

  JsonObject:       Mapping[str, Any] — opaque. Skips value validation.
                    For data already parsed from JSON (Chrome Local State,
                    CDP responses, MCP tool inputs).

  StrictJsonObject: Mapping[str, JsonValue] — validated. Rejects non-JSON
                    values (set, bytes, datetime) via recursive checking.
                    For constructing JSON to send where provenance is uncertain.

  pydantic.JsonValue (import directly from pydantic):
                    Any JSON value — str, int, float, bool, list, dict, None.
                    For fields that could be any JSON type, not just objects.

  Performance: JsonObject is instant (Any skips validation). StrictJsonObject
  and JsonValue validate recursively — negligible for typical payloads, but
  ~130x slower than Any for large nested structures in Python mode. JSON mode
  (model_validate_json) has no overhead for any variant.

  Prefer specific Pydantic models when the JSON structure is known.

JSON serialization helpers — Pydantic strict mode rejects string-to-type
coercion by default, but JSON always encodes datetimes and UUIDs as strings:

  JsonDatetime:     datetime that accepts ISO string input (strict=False).
  JsonUuid:         UUID that accepts string input (strict=False).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import pydantic
from pydantic import JsonValue

# -- Session lifecycle --------------------------------------------------------

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']

# -- Claude Code versioning ---------------------------------------------------

type CCVersion = str

# -- JSON serialization helpers -----------------------------------------------

JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
JsonUuid = Annotated[UUID, pydantic.Field(strict=False)]

# -- JSON value types ---------------------------------------------------------

type JsonObject = Mapping[str, Any]
type StrictJsonObject = Mapping[str, JsonValue]
