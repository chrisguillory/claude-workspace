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

__all__ = [
    'CCVersion',
    'EffortLevel',
    'JsonDatetime',
    'JsonObject',
    'JsonUuid',
    'OutputFormat',
    'PydanticVersion',
    'SessionSource',
    'SessionState',
    'StrictJsonObject',
]

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal, Self
from uuid import UUID

import pydantic
from packaging.version import Version
from pydantic import GetCoreSchemaHandler, JsonValue
from pydantic_core import CoreSchema, core_schema

# -- Session lifecycle --------------------------------------------------------

type SessionState = Literal['active', 'exited', 'completed', 'crashed']
type SessionSource = Literal['startup', 'resume', 'compact', 'clear']

# -- Reasoning effort ---------------------------------------------------------

type EffortLevel = Literal['low', 'medium', 'high', 'xhigh', 'max']

# -- CLI output format --------------------------------------------------------

type OutputFormat = Literal['text', 'json']

# -- Claude Code versioning ---------------------------------------------------


class PydanticVersion(Version):
    """A PEP 440 ``Version`` usable as a Pydantic field — validates from a string, serializes to one.

    Subclasses inherit the Pydantic plumbing and override ``parse`` to accept a
    looser raw form (see ``CCVersion``). Comparison and hashing follow PEP 440
    version order; ``.major`` / ``.minor`` / ``.micro`` are inherited from
    ``packaging.version.Version``.
    """

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse a raw version string; the base implementation trims surrounding whitespace."""
        return cls(raw.strip())

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,  # noqa: ARG003 — required by Pydantic protocol
        handler: GetCoreSchemaHandler,  # noqa: ARG003 — required by Pydantic protocol
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str, when_used='json'),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,  # noqa: ARG003 — required by Pydantic protocol
        handler: pydantic.GetJsonSchemaHandler,  # noqa: ARG003 — required by Pydantic protocol
    ) -> Mapping[str, Any]:
        return {'type': 'string', 'description': 'PEP 440 version string (e.g., "1.2.3")'}

    @classmethod
    def _pydantic_validate(cls, value: object) -> Self:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        # Pydantic wraps ValueError into ValidationError; TypeError would leak through.
        raise ValueError(f'Cannot construct {cls.__name__} from {type(value).__name__}')


class CCVersion(PydanticVersion):
    """Claude Code version — a PEP 440 ``Version`` with a relaxed parser.

    Direct construction (``CCVersion('2.1.131')``) requires strict PEP 440
    input. Use ``CCVersion.parse(raw)`` to handle ``claude --version`` output
    (``'2.1.131 (Claude Code)'``) — it strips the ``' (Claude Code)'`` suffix
    before parsing.

    Comparison and hashing follow PEP 440 version order
    (``CCVersion('2.1.10') > CCVersion('2.1.9')``); ``.major`` / ``.minor`` /
    ``.micro`` are inherited. Pydantic models serialize as a string and
    deserialize via ``parse``.
    """

    @classmethod
    def parse(cls, raw: str) -> Self:
        """Parse raw ``claude --version`` output, stripping the ``(Claude Code)`` suffix."""
        stripped = raw.strip()
        cleaned = stripped.split()[0] if stripped else stripped
        return cls(cleaned)


# -- JSON serialization helpers -----------------------------------------------

JsonDatetime = Annotated[datetime, pydantic.Field(strict=False)]
JsonUuid = Annotated[UUID, pydantic.Field(strict=False)]

# -- JSON value types ---------------------------------------------------------

type JsonObject = Mapping[str, Any]
type StrictJsonObject = Mapping[str, JsonValue]
