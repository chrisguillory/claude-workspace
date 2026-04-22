"""Length-prefixed JSON framing over TCP."""

from __future__ import annotations

import asyncio
import json
import struct

import pydantic

from claude_remote_bash.models import Message

__all__ = [
    'ProtocolError',
    'read_message',
    'write_message',
]

# Wire format: [4 bytes uint32 BE length] [1 byte flags] [N bytes JSON]
# Flags byte reserved for future compression (bit 0: compressed, bits 1-2: algorithm).
HEADER_FORMAT = '!IB'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10 MB guard

_MESSAGE_ADAPTER: pydantic.TypeAdapter[Message] = pydantic.TypeAdapter(Message)


class ProtocolError(Exception):
    """Protocol-level error (framing, size limits, malformed JSON)."""


async def read_message(reader: asyncio.StreamReader) -> Message:
    """Read and validate a length-prefixed JSON message from the stream.

    The raw JSON is deserialized into the appropriate Pydantic model via
    discriminated union on the ``type`` field.

    Raises:
        ProtocolError: On framing errors, oversized payloads, or malformed JSON.
        pydantic.ValidationError: On schema validation failure.
        asyncio.IncompleteReadError: On unexpected connection close.
    """
    header = await reader.readexactly(HEADER_SIZE)
    length, flags = struct.unpack(HEADER_FORMAT, header)

    if flags != 0:
        raise ProtocolError(f'unsupported flags byte: 0x{flags:02x}')

    if length > MAX_PAYLOAD_SIZE:
        raise ProtocolError(f'payload too large: {length} bytes (max {MAX_PAYLOAD_SIZE})')

    payload = await reader.readexactly(length)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ProtocolError(f'malformed JSON: {e}') from e

    if not isinstance(data, dict):
        raise ProtocolError(f'expected JSON object, got {type(data).__name__}')

    return _MESSAGE_ADAPTER.validate_python(data)


async def write_message(writer: asyncio.StreamWriter, msg: Message) -> None:
    """Serialize and write a length-prefixed JSON message to the stream."""
    payload = _MESSAGE_ADAPTER.dump_json(msg, by_alias=True)
    header = struct.pack(HEADER_FORMAT, len(payload), 0x00)
    writer.write(header + payload)
    await writer.drain()
